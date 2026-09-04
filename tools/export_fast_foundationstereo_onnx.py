#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export a fixed-batch Fast-FoundationStereo ONNX artifact.

The model implementation remains in an external Fast-FoundationStereo checkout.
This wrapper reuses its ONNX-compatible model wrapper while making the static
batch size and artifact provenance explicit.
"""

import argparse
import hashlib
import logging
import os
from pathlib import Path
import runpy
import subprocess
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fixed-batch Fast-FoundationStereo ONNX artifact"
    )
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Fast-FoundationStereo checkout")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Official serialized .pth checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--onnx-name")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.height % 32 or args.width % 32:
        parser.error("--height and --width must be divisible by 32")
    return args


def main() -> None:
    args = _arguments()
    source_dir = args.source_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    exporter = source_dir / "scripts" / "make_single_onnx.py"
    if not exporter.is_file():
        raise FileNotFoundError(f"Fast-FoundationStereo exporter not found: {exporter}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Make the external checkout's `core` package win over this workspace's core.
    sys.path.insert(0, str(source_dir))
    upstream = runpy.run_path(str(exporter), run_name="_ffs_onnx_exporter")

    import torch
    import yaml
    from omegaconf import OmegaConf

    model_class = upstream["FastFoundationStereoSingleOnnx"]
    foundation_stereo = upstream["_fs_module"]
    foundation_stereo.normalize_image = lambda image: image
    foundation_stereo.build_gwc_volume_optimized_pytorch1 = upstream[
        "_build_gwc_volume_onnx"
    ]
    foundation_stereo.build_concat_volume_optimized_pytorch1 = upstream[
        "_build_concat_volume_onnx"
    ]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.autograd.set_grad_enabled(False)
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.args.mixed_precision = False
    model.cuda().eval()
    wrapper = model_class(model).cuda().eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_name = args.onnx_name or (
        f"fast_foundationstereo_b{args.batch_size}_{args.height}x{args.width}"
    )
    onnx_name = onnx_name if onnx_name.endswith(".onnx") else f"{onnx_name}.onnx"
    onnx_path = args.output_dir / onnx_name
    images = (
        torch.randn(args.batch_size, 3, args.height, args.width, device="cuda"),
        torch.randn(args.batch_size, 3, args.height, args.width, device="cuda"),
    )
    logging.info("Exporting static batch %d to %s", args.batch_size, onnx_path)
    torch.onnx.export(
        wrapper,
        images,
        onnx_path,
        opset_version=17,
        input_names=["left_image", "right_image"],
        output_names=["disparity"],
        do_constant_folding=True,
        dynamo=False,
    )

    metadata = OmegaConf.to_container(model.args, resolve=True)
    metadata.update({
        "batch_size": args.batch_size,
        "image_size": [args.height, args.width],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "upstream_revision": _git_revision(source_dir),
        "exporter_revision": _git_revision(Path(__file__).resolve().parent),
    })
    metadata_path = onnx_path.with_suffix(".yaml")
    with metadata_path.open("w") as file:
        yaml.safe_dump(metadata, file, sort_keys=True)
    logging.info("ONNX: %s", onnx_path)
    logging.info("Metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
