"""Utility functions for plotting and tensor/array conversion used by
PartitionManagers experiments.

This module provides helpers to convert Pytorch tensors (or lists) to
NumPy arrays and to save a simple scatter plot comparing ground-truth vs
predicted values.
"""

import threading

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    # nvidia-ml-py exposes the same API under the `pynvml` module name.
    # Prefer it over the deprecated standalone `pynvml` package.
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
        nvmlShutdown,
    )

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GPURAMStepSampler:  # pylint: disable=too-many-instance-attributes
    """Sample GPU and host RAM telemetry periodically and provide per-step aggregates."""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        enabled,
        sample_interval_sec,
        logger,
        device_index=0,
        enable_gpu=True,
        enable_ram=True,
    ):
        """Initialize the telemetry sampler.

        Args:
            enabled: Whether sampling is enabled.
            sample_interval_sec: Sampling period in seconds.
            logger: Logger instance used for status messages.
            device_index: CUDA device index used by NVML.
            enable_gpu: Whether to collect GPU memory samples.
            enable_ram: Whether to collect host RAM samples.
        """
        self.enabled = bool(enabled)
        self.sample_interval_sec = max(float(sample_interval_sec), 0.01)
        self.logger = logger
        self.device_index = int(device_index)
        self.enable_gpu = bool(enable_gpu)
        self.enable_ram = bool(enable_ram)

        self._handle = None
        self._is_ready = False
        self._gpu_ready = False
        self._ram_ready = False
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._sample_count = 0
        self._memory_used_mb_sum = 0.0
        self._memory_used_mb_sq_sum = 0.0
        self._ram_used_mb_sum = 0.0
        self._ram_used_mb_sq_sum = 0.0

        if not self.enabled:
            return

        self._initialize_gpu_monitor()
        self._initialize_ram_monitor()

        self._is_ready = self._gpu_ready or self._ram_ready
        if not self._is_ready:
            self.enabled = False

    def _initialize_gpu_monitor(self):
        """Initialize NVML-based GPU monitoring if available and enabled."""
        if not self.enable_gpu:
            return

        if not torch.cuda.is_available():
            self.logger.info("GPU monitor disabled: CUDA is not available.")
            return

        if not NVML_AVAILABLE:
            self.logger.warning(
                "GPU monitor enabled but `pynvml` is not installed. "
                "Install it with: pip install nvidia-ml-py"
            )
            return

        try:
            nvmlInit()
            self._handle = nvmlDeviceGetHandleByIndex(self.device_index)
            self._gpu_ready = True

            device_name = nvmlDeviceGetName(self._handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode("utf-8", errors="replace")
            mem_info = nvmlDeviceGetMemoryInfo(self._handle)
            total_mem_gb = mem_info.total / (1024.0**3)

            self.logger.info(
                "GPU monitor enabled | device=%d | name=%s | total_memory=%.2f GiB | sample_every=%.3fs",
                self.device_index,
                device_name,
                total_mem_gb,
                self.sample_interval_sec,
            )
        except Exception as exc:
            self.logger.warning("Could not initialize NVML. GPU monitor disabled: %s", exc)

    def _initialize_ram_monitor(self):
        """Initialize psutil-based RAM monitoring if available and enabled."""
        if not self.enable_ram:
            return

        if not PSUTIL_AVAILABLE:
            self.logger.warning(
                "RAM monitor enabled but `psutil` is not installed. "
                "Install it with: pip install psutil"
            )
            return

        try:
            vm = psutil.virtual_memory()
            total_ram_gb = vm.total / (1024.0**3)
            self._ram_ready = True
            self.logger.info(
                "RAM monitor enabled | total_memory=%.2f GiB | sample_every=%.3fs",
                total_ram_gb,
                self.sample_interval_sec,
            )
        except Exception as exc:
            self.logger.warning("Could not initialize RAM monitor. RAM monitor disabled: %s", exc)

    def _reset_buffers(self):
        """Reset per-step sample counters and accumulators."""
        self._sample_count = 0
        self._memory_used_mb_sum = 0.0
        self._memory_used_mb_sq_sum = 0.0
        self._ram_used_mb_sum = 0.0
        self._ram_used_mb_sq_sum = 0.0

    @staticmethod
    def _mean(total, count):
        """Return the arithmetic mean for ``total`` and ``count``."""
        return float(total / count) if count > 0 else 0.0

    @staticmethod
    def _var(total, sq_total, count):
        """Return population variance from running sum and squared-sum."""
        if count <= 0:
            return 0.0
        mean = total / count
        variance = (sq_total / count) - (mean * mean)
        return float(max(variance, 0.0))

    def _sample_once(self):
        """Take one telemetry sample and update shared accumulators."""
        with self._lock:
            self._sample_count += 1
            if self._gpu_ready:
                mem_info = nvmlDeviceGetMemoryInfo(self._handle)
                mem_used_mb = mem_info.used / (1024.0**2)
                self._memory_used_mb_sum += float(mem_used_mb)
                self._memory_used_mb_sq_sum += float(mem_used_mb * mem_used_mb)

            if self._ram_ready:
                ram_info = psutil.virtual_memory()
                ram_used_mb = ram_info.used / (1024.0**2)
                self._ram_used_mb_sum += float(ram_used_mb)
                self._ram_used_mb_sq_sum += float(ram_used_mb * ram_used_mb)

    def _sampling_loop(self):
        """Run the periodic sampling loop until a stop signal is received."""
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception:
                pass
            self._stop_event.wait(self.sample_interval_sec)

    def start(self):
        """Start background sampling and reset current buffers."""
        if not (self.enabled and self._is_ready):
            return

        self.stop()
        with self._lock:
            self._reset_buffers()

        self._stop_event.clear()

        # Take one immediate sample to avoid zero-sample summaries when
        # short/blocked sections prevent the background thread from running.
        try:
            self._sample_once()
        except Exception:
            pass

        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the sampling thread and wait briefly for its termination."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.sample_interval_sec * 4))
        self._thread = None

    def summary(self):
        """Return aggregated telemetry statistics for the current step.

        Returns:
            dict[str, float | int]: Dictionary with sample count, mean, and
            variance for GPU and RAM memory usage.
        """
        if not (self.enabled and self._is_ready):
            return {
                "gpu_samples": 0,
                "gpu_mem_used_mb_mean": 0.0,
                "gpu_mem_used_mb_var": 0.0,
                "ram_samples": 0,
                "ram_used_mb_mean": 0.0,
                "ram_used_mb_var": 0.0,
            }

        with self._lock:
            sample_count = self._sample_count
            mem_used_mb_sum = self._memory_used_mb_sum
            mem_used_mb_sq_sum = self._memory_used_mb_sq_sum
            ram_used_mb_sum = self._ram_used_mb_sum
            ram_used_mb_sq_sum = self._ram_used_mb_sq_sum

        if sample_count == 0:
            try:
                self._sample_once()
            except Exception:
                pass
            with self._lock:
                sample_count = self._sample_count
                mem_used_mb_sum = self._memory_used_mb_sum
                mem_used_mb_sq_sum = self._memory_used_mb_sq_sum
                ram_used_mb_sum = self._ram_used_mb_sum
                ram_used_mb_sq_sum = self._ram_used_mb_sq_sum

        return {
            "gpu_samples": sample_count if self._gpu_ready else 0,
            "gpu_mem_used_mb_mean": self._mean(mem_used_mb_sum, sample_count) if self._gpu_ready else 0.0,
            "gpu_mem_used_mb_var": self._var(mem_used_mb_sum, mem_used_mb_sq_sum, sample_count) if self._gpu_ready else 0.0,
            "ram_samples": sample_count if self._ram_ready else 0,
            "ram_used_mb_mean": self._mean(ram_used_mb_sum, sample_count) if self._ram_ready else 0.0,
            "ram_used_mb_var": self._var(ram_used_mb_sum, ram_used_mb_sq_sum, sample_count) if self._ram_ready else 0.0,
        }

    def close(self):
        """Release monitor resources and disable future sampling."""
        self.stop()
        if self._gpu_ready:
            try:
                nvmlShutdown()
            except Exception:
                pass
            self._gpu_ready = False
        self._ram_ready = False
        self._is_ready = False



def to_numpy(data):
    """Convertir ``data`` a un ndarray de NumPy.

    - Si ``data`` es un ``torch.Tensor`` se mueve a CPU y se desconecta del
      grafo para obtener un array.
    - Si ``data`` es una ``list`` se convierte con ``np.array``.
    - En cualquier otro caso se devuelve ``data`` tal cual.

    Args:

    data: ``torch.Tensor``, ``list`` u objeto ya en formato NumPy.

    Returns:
        Un objeto NumPy (o el valor original si no aplica conversión).
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        return np.array(data)
    return data

def plot_multi_method_comparison(
    dataset,
    predictions_dict,
    dim,
    title,
    outpath,
):  # pylint: disable=too-many-locals
    """Comparación visual 3D entre Ground Truth y métodos.

    Crea una figura con subplots: uno para los datos reales y uno por cada
    método en ``predictions_dict``. Cada subplot muestra los puntos de prueba
    y de entrenamiento, y en el caso de los métodos se reporta el MSE local.

    Parameters
    ----------
    dataset : dict
        Diccionario con los datos base: 'x_test', 'y_test', 'x_train', 'y_train'.

    predictions_dict : dict[str, array-like or torch.Tensor]
        Diccionario con las predicciones por método. Llaves son nombres de
        método y valores son tensores/arrays con las predicciones sobre
        ``x_test``.

    dim : int
        Dimensionalidad del problema. Actualmente solo se grafica si ``dim == 2``.

    title : str
        Título base para los subplots.

    outpath : str
        Ruta de salida del archivo PNG que se genera.
    """

    # Filtro: Solo graficamos Dim 2 (Superficie)
    if dim != 2:
        return

    x_test = dataset["x_test"]
    y_test = dataset["y_test"]
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]

    # Convertir datos base a numpy
    xt = to_numpy(x_test)
    yt = to_numpy(y_test).flatten()
    xtr = to_numpy(x_train)
    ytr = to_numpy(y_train).flatten()

    # --- NUEVO: Calcular límites globales basados en Ground Truth y Train ---
    # Esto asegura que todos los plots tengan la misma escala vertical.
    z_min = min(np.min(yt), np.min(ytr))
    z_max = max(np.max(yt), np.max(ytr))

    # Opcional: Agregar un pequeño margen (padding) del 5% para que los puntos no toquen los bordes
    z_range = z_max - z_min
    z_min -= z_range * 0.05
    z_max += z_range * 0.05

    methods = list(predictions_dict.keys())
    n_methods = len(methods)

    # Configuración de la figura: 1 fila, 1 (GT) + n_methods columnas
    cols = 1 + n_methods
    fig = plt.figure(figsize=(6 * cols, 6))

    # --- SUBPLOT 1: GROUND TRUTH ---
    ax1 = fig.add_subplot(1, cols, 1, projection='3d')
    # Puntos reales (Gris)
    ax1.scatter(
        xt[:, 0],
        xt[:, 1],
        yt,
        c='0.4',
        marker='.',
        s=15,
        alpha=0.2,
        label='Ground Truth'
    )

    # Puntos de entrenamiento (Rojos)


    ax1.scatter(
        xtr[:, 0],
        xtr[:, 1],
        ytr,
        c='r',
        marker='x',
        s=40,
        label='Train Data'
    )

    ax1.set_title(f"Ground Truth\n{title}")
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('Y')

    # APLICAR ESCALA
    ax1.set_zlim(z_min, z_max)

    ax1.view_init(elev=30, azim=-60)

    # --- SUBPLOTS MÉTODOS ---
    # Colores para diferenciar métodos: Azul, Verde, Purpura...
    colors = ['b', 'g', 'm', 'c']

    for i, method_name in enumerate(methods):
        y_pred_tensor = predictions_dict[method_name]
        yp = to_numpy(y_pred_tensor).flatten()

        # Calcular MSE localmente para el título
        mse_val = np.mean((yt - yp)**2)

        ax = fig.add_subplot(1, cols, i + 2, projection='3d')

        # Predicción
        col = colors[i % len(colors)]
        ax.scatter(
            xt[:, 0],
            xt[:, 1],
            yp,
            c=col,
            marker='.',
            s=15,
            alpha=0.2,
            label='Predicción'
        )

        # Referencia Entrenamiento (para ver si pasaron por los puntos)
        ax.scatter(xtr[:, 0], xtr[:, 1], ytr, c='r', marker='x', s=40)

        ax.set_title(f"Modelo: {method_name.upper()}\nMSE: {mse_val:.5f}")
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')

        # APLICAR ESCALA (La misma del GT)
        ax.set_zlim(z_min, z_max)

        ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()

def quadratic_steps(dim, max_points=1000, n_steps=5, max_dim=10):
    steps = []
    for i in range(1, n_steps + 1):
        val = max_points * (i / n_steps) ** 2 * (dim / max_dim)
        steps.append(int(val))
    return steps