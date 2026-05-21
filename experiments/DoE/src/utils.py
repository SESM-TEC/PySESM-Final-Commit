"""Telemetría (GPU + RAM) y helpers de I/O para el experimento Ring Oscillator.

Es una versión slim del módulo equivalente en PartitionManagersExp, solo
incluyendo lo necesario para este experimento (no hay plotting 2D).
"""

import csv
import os
import threading
import time

import numpy as np
import torch

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlInit,
        nvmlShutdown,
    )
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GPURAMStepSampler:  # pylint: disable=too-many-instance-attributes
    """Muestreo periódico de memoria GPU (NVML) y RAM (psutil)."""

    def __init__(self, enabled, sample_interval_sec, logger,
                 device_index=0, enable_gpu=True, enable_ram=True):
        self.enabled = bool(enabled)
        self.sample_interval_sec = max(float(sample_interval_sec), 0.01)
        self.logger = logger
        self.device_index = int(device_index)
        self.enable_gpu = bool(enable_gpu)
        self.enable_ram = bool(enable_ram)

        self._handle = None
        self._gpu_ready = False
        self._ram_ready = False
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._reset_buffers()

        if not self.enabled:
            return

        self._init_gpu()
        self._init_ram()
        self._is_ready = self._gpu_ready or self._ram_ready
        if not self._is_ready:
            self.enabled = False

    def _init_gpu(self):
        if not self.enable_gpu or not torch.cuda.is_available() or not NVML_AVAILABLE:
            return
        try:
            nvmlInit()
            self._handle = nvmlDeviceGetHandleByIndex(self.device_index)
            self._gpu_ready = True
            name = nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = nvmlDeviceGetMemoryInfo(self._handle)
            self.logger.info("GPU monitor: %s | total=%.2f GiB | every=%.3fs",
                             name, mem.total / 1024**3, self.sample_interval_sec)
        except Exception as exc:
            self.logger.warning("NVML init failed: %s", exc)

    def _init_ram(self):
        if not self.enable_ram or not PSUTIL_AVAILABLE:
            return
        try:
            vm = psutil.virtual_memory()
            self._ram_ready = True
            self.logger.info("RAM monitor: total=%.2f GiB | every=%.3fs",
                             vm.total / 1024**3, self.sample_interval_sec)
        except Exception as exc:
            self.logger.warning("psutil init failed: %s", exc)

    def _reset_buffers(self):
        self._sample_count = 0
        self._gpu_sum = 0.0
        self._gpu_sq = 0.0
        self._ram_sum = 0.0
        self._ram_sq = 0.0

    @staticmethod
    def _mean(total, count):
        return float(total / count) if count > 0 else 0.0

    @staticmethod
    def _var(total, sq, count):
        if count <= 0:
            return 0.0
        mean = total / count
        return float(max(sq / count - mean * mean, 0.0))

    def _sample_once(self):
        with self._lock:
            self._sample_count += 1
            if self._gpu_ready:
                mb = nvmlDeviceGetMemoryInfo(self._handle).used / 1024**2
                self._gpu_sum += mb
                self._gpu_sq += mb * mb
            if self._ram_ready:
                mb = psutil.virtual_memory().used / 1024**2
                self._ram_sum += mb
                self._ram_sq += mb * mb

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception:
                pass
            self._stop_event.wait(self.sample_interval_sec)

    def start(self):
        if not (self.enabled and (self._gpu_ready or self._ram_ready)):
            return
        self.stop()
        with self._lock:
            self._reset_buffers()
        self._stop_event.clear()
        try:
            self._sample_once()
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.sample_interval_sec * 4))
        self._thread = None

    def summary(self):
        if not (self.enabled and (self._gpu_ready or self._ram_ready)):
            return {"gpu_samples": 0, "gpu_mem_used_mb_mean": 0.0,
                    "gpu_mem_used_mb_var": 0.0, "ram_samples": 0,
                    "ram_used_mb_mean": 0.0, "ram_used_mb_var": 0.0}
        with self._lock:
            n, gs, gsq, rs, rsq = (self._sample_count, self._gpu_sum,
                                   self._gpu_sq, self._ram_sum, self._ram_sq)
        if n == 0:
            try:
                self._sample_once()
            except Exception:
                pass
            with self._lock:
                n, gs, gsq, rs, rsq = (self._sample_count, self._gpu_sum,
                                       self._gpu_sq, self._ram_sum, self._ram_sq)
        return {
            "gpu_samples": n if self._gpu_ready else 0,
            "gpu_mem_used_mb_mean": self._mean(gs, n) if self._gpu_ready else 0.0,
            "gpu_mem_used_mb_var":  self._var(gs, gsq, n) if self._gpu_ready else 0.0,
            "ram_samples": n if self._ram_ready else 0,
            "ram_used_mb_mean": self._mean(rs, n) if self._ram_ready else 0.0,
            "ram_used_mb_var":  self._var(rs, rsq, n) if self._ram_ready else 0.0,
        }

    def close(self):
        self.stop()
        if self._gpu_ready:
            try:
                nvmlShutdown()
            except Exception:
                pass
            self._gpu_ready = False
        self._ram_ready = False


# ---------------- I/O helpers ------------------------------

RESULT_FIELDS = [
    'run_id', 'target', 'method', 'n_samples',
    'mse_norm', 'mae_norm', 'mse_orig', 'mae_orig',
    'train_time', 'test_time',
    'gpu_samples', 'gpu_mem_used_mb_mean', 'gpu_mem_used_mb_var',
    'torch_peak_alloc_mb', 'torch_peak_reserved_mb',
    'ram_samples', 'ram_used_mb_mean', 'ram_used_mb_var',
    'timestamp',
]


def load_existing_results(filepath):
    """Devuelve un set con llaves (run_id, target, method, n_samples) ya escritas."""
    done = set()
    if not os.path.exists(filepath):
        return done
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((int(row['run_id']), row['target'],
                          row['method'], int(row['n_samples'])))
    except Exception as exc:
        print(f"[checkpoint] no se pudo leer {filepath}: {exc}")
    return done


def save_result_row(filepath, data_dict):
    """Append-only escritor (escribe header en la primera vez)."""
    exists = os.path.exists(filepath)
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            if not exists:
                writer.writeheader()
            row = {k: data_dict.get(k, '') for k in RESULT_FIELDS}
            row['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(row)
    except Exception as exc:
        print(f"[checkpoint] error escribiendo {filepath}: {exc}")


# ---------------- Normalización ----------------------------

def fit_feature_scaler(ranges, feature_columns):
    """Devuelve (mins, maxs) torch tensors a partir de los rangos del YAML."""
    mins = torch.tensor([float(ranges[c][0]) for c in feature_columns], dtype=torch.float32)
    maxs = torch.tensor([float(ranges[c][1]) for c in feature_columns], dtype=torch.float32)
    return mins, maxs


def scale_features(x_raw, mins, maxs):
    """Min-max scaling a [0, 1] usando rangos del YAML (no del dato)."""
    return (x_raw - mins) / (maxs - mins).clamp(min=1e-30)


class TargetScaler:
    """Pipeline: opcional log10 → standardize (fit con datos de entrenamiento)."""

    def __init__(self, use_log=True):
        self.use_log = bool(use_log)
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y_raw: torch.Tensor):
        y = y_raw.clone().float()
        if self.use_log:
            # Evita log(0) o negativos descartando esos valores en el fit.
            y = y[y > 0]
            y = torch.log10(y)
        self.mean = float(y.mean().item())
        self.std = float(y.std().item()) or 1.0
        return self

    def transform(self, y_raw: torch.Tensor) -> torch.Tensor:
        y = y_raw.clone().float()
        if self.use_log:
            y = torch.log10(y.clamp(min=1e-30))
        return (y - self.mean) / self.std

    def inverse(self, y_norm: torch.Tensor) -> torch.Tensor:
        y = y_norm * self.std + self.mean
        if self.use_log:
            y = torch.pow(10.0, y)
        return y
