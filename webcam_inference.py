"""Run FlashDepth inference on live webcam frames."""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from dataloaders.depthanything_preprocess import depthanything_preprocess
from flashdepth.model import FlashDepth
from utils.helpers import depth_to_np_arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashDepth webcam demo")
    parser.add_argument(
        "--config-path",
        default="configs/flashdepth",
        help="Path to a config directory or YAML file (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint to load (iter_xxxxx.pth or state dict)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device used for inference (default: %(default)s)",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Computation precision when running on CUDA (default: %(default)s)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default: %(default)s)",
    )
    parser.add_argument(
        "--short-side",
        type=int,
        default=518,
        help="Resize webcam frames so the short side matches this value (default: %(default)s)",
    )
    parser.add_argument(
        "--enable-hybrid",
        action="store_true",
        help="Use the hybrid teacher (requires --teacher-checkpoint)",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="Teacher checkpoint for hybrid fusion (FlashDepth-L)",
    )
    parser.add_argument(
        "--window-name",
        default="FlashDepth Webcam",
        help="Name of the OpenCV preview window",
    )
    parser.add_argument(
        "--max-fps-samples",
        type=int,
        default=30,
        help="Number of frames used for the rolling FPS average (default: %(default)s)",
    )
    return parser.parse_args()


def load_config(config_path: str):
    cfg_path = config_path
    if os.path.isdir(config_path):
        cfg_path = os.path.join(config_path, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    cfg.inference = True
    cfg.training.batch_size = 1
    cfg.eval.compile = False
    return cfg


def build_model(
    cfg,
    checkpoint_path: str,
    device: torch.device,
    enable_hybrid: bool,
    teacher_checkpoint: str | None,
) -> FlashDepth:
    if not enable_hybrid:
        cfg.hybrid_configs.use_hybrid = False
    elif teacher_checkpoint is None:
        raise ValueError("--enable-hybrid requires --teacher-checkpoint to be set")
    else:
        cfg.hybrid_configs.teacher_model_path = teacher_checkpoint
        cfg.hybrid_configs.use_hybrid = True

    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model = FlashDepth(
        batch_size=1,
        hybrid_configs=cfg.hybrid_configs,
        training=False,
        **model_kwargs,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected}")

    model.eval()
    model.to(device)
    if hasattr(model, "mamba") and model.use_mamba:
        model.mamba.start_new_sequence()
    return model


def make_autocast(device: torch.device, precision: torch.dtype) -> Callable[[], contextlib.AbstractContextManager]:
    if device.type == "cuda" and precision in (torch.float16, torch.bfloat16):
        return lambda: torch.autocast(device_type="cuda", dtype=precision)
    return contextlib.nullcontext


def preprocess_frame(frame: np.ndarray, short_side: int) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    scale = short_side / min(h, w)
    target_h = int(round(h * scale))
    target_w = int(round(w * scale))
    tensor = depthanything_preprocess(rgb, width=target_w, height=target_h)
    return tensor.float()


@torch.no_grad()
def predict_depth(
    model: FlashDepth,
    frame_tensor: torch.Tensor,
    amp_ctx: Callable[[], contextlib.AbstractContextManager],
) -> torch.Tensor:
    if frame_tensor.dim() == 3:
        frame_tensor = frame_tensor.unsqueeze(0)
    frame_tensor = frame_tensor.to(next(model.parameters()).device, non_blocking=True)

    with amp_ctx():
        B, C, H, W = frame_tensor.shape
        patch_h = H // model.patch_size
        patch_w = W // model.patch_size
        dpt_features = model.get_dpt_features(frame_tensor, input_shape=(B, C, H, W))
        pred_depth = model.final_head(dpt_features, patch_h, patch_w)
        pred_depth = torch.clamp(pred_depth, min=0)
    return pred_depth


def depth_to_colormap(depth: torch.Tensor, target_hw: tuple[int, int]) -> np.ndarray:
    colorized = depth_to_np_arr(depth)[0]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
    return cv2.resize(colorized, target_hw, interpolation=cv2.INTER_LINEAR)


def main():
    args = parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA requested but not available. Falling back to CPU.")

    cfg = load_config(args.config_path)
    precision_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    requested_precision = precision_map[args.precision]
    if device.type != "cuda":
        requested_precision = torch.float32

    model = build_model(cfg, args.checkpoint, device, args.enable_hybrid, args.teacher_checkpoint)
    amp_ctx = make_autocast(device, requested_precision)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    fps_window = deque(maxlen=args.max_fps_samples)

    try:
        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                print("[WARN] Failed to read from camera. Retrying...")
                time.sleep(0.05)
                continue

            input_tensor = preprocess_frame(frame, args.short_side)
            start_time = time.perf_counter()
            depth = predict_depth(model, input_tensor, amp_ctx)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fps_window.append(1.0 / max(time.perf_counter() - start_time, 1e-6))

            depth_vis = depth_to_colormap(depth, (frame.shape[1], frame.shape[0]))
            stacked = np.hstack([frame, depth_vis])
            fps_text = f"{np.mean(fps_window):.1f} FPS" if fps_window else "-- FPS"
            cv2.putText(
                stacked,
                f"FlashDepth | {fps_text}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                thickness=4,
            )
            cv2.putText(
                stacked,
                f"FlashDepth | {fps_text}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                thickness=2,
            )
            cv2.imshow(args.window_name, stacked)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
