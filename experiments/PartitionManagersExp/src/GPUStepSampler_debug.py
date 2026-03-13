"""Debug harness for GPUStepSampler.

Use this script to validate whether NVML sampling is working and to inspect
why `GPU_Samples` might be zero in experiment runs.

Examples:
    python src/GPUStepSampler_debug.py
    python src/GPUStepSampler_debug.py --duration 10 --sample-interval 0.2
    python src/GPUStepSampler_debug.py --duration 5 --no-workload
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch

from src.utils import GPUStepSampler


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("GPUStepSamplerDebug")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def _torch_gpu_info() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if info["cuda_available"]:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        info.update(
            {
                "device_index": device_index,
                "device_name": torch.cuda.get_device_name(device_index),
                "total_memory_gib": round(props.total_memory / (1024**3), 3),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return info


def _run_torch_workload(duration_sec: float, device: torch.device) -> int:
    start = time.time()
    iterations = 0
    matrix_a = torch.randn((2048, 2048), device=device)
    matrix_b = torch.randn((2048, 2048), device=device)

    while (time.time() - start) < duration_sec:
        result = matrix_a @ matrix_b
        result = torch.relu(result)
        del result
        iterations += 1

    torch.cuda.synchronize(device)
    return iterations


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug GPUStepSampler behavior.")
    parser.add_argument("--duration", type=float, default=5.0, help="Sampling duration in seconds.")
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="Sampler interval in seconds.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--no-workload",
        action="store_true",
        help="Do not run synthetic GPU workload while sampling.",
    )
    args = parser.parse_args()

    logger = _build_logger()
    logger.info("Starting GPUStepSampler debug script")

    torch_info = _torch_gpu_info()
    logger.info("Torch GPU info: %s", json.dumps(torch_info, ensure_ascii=False))

    sampler = GPUStepSampler(
        enabled=True,
        sample_interval_sec=args.sample_interval,
        logger=logger,
        device_index=args.device,
    )

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Sampler cannot collect GPU telemetry.")
        summary = sampler.summary()
        logger.info("Sampler summary: %s", json.dumps(summary, ensure_ascii=False))
        sampler.close()
        return 0

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    sampler.start()
    start_time = time.time()

    if args.no_workload:
        logger.info("No-workload mode: sleeping for %.2fs", args.duration)
        time.sleep(args.duration)
        iterations = 0
    else:
        logger.info("Running synthetic GPU workload for %.2fs", args.duration)
        iterations = _run_torch_workload(args.duration, device)

    elapsed = time.time() - start_time
    sampler.stop()
    summary = sampler.summary()

    logger.info("Workload iterations: %d", iterations)
    logger.info("Elapsed: %.3fs", elapsed)
    logger.info("Sampler summary: %s", json.dumps(summary, ensure_ascii=False))

    sampler.close()

    if summary.get("gpu_samples", 0) <= 0:
        logger.error("No GPU samples captured (gpu_samples=0).")
        return 2

    logger.info("GPU sampling looks healthy (gpu_samples > 0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
