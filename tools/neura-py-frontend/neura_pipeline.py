#!/usr/bin/env python3
"""
neura_pipeline.py — Python Frontend Lowering Pipeline to Neura Dialect

This script provides the complete lowering path from PyTorch models to the
Neura CGRA dataflow dialect, executed in three stages:

    PyTorch model
      → torch-mlir (Linalg-on-Tensors IR)
        → Stage 1: --linalg-to-affine-conversion  (→ Affine)
        → Stage 2: --taskflow-conversion           (→ Neura)
        → Stage 3: --neura-conversion              (→ Neura optimised + Codegen)
          → CGRA JSON instructions

All three stages run through mlir-neura-opt. The final output is fully
lowered to the Neura dialect with code generation.

Usage:
    # Generate MLIR from a PyTorch model and lower through the full pipeline
    python neura_pipeline.py --model model.py --output out.json

    # Or pass an existing Linalg-on-Tensors .mlir file through the pipeline
    python neura_pipeline.py --input linalg_ir.mlir --output out.json

    # Stop after a specific stage (intermediate output is still Neura dialect)
    python neura_pipeline.py --input linalg_ir.mlir --output out.mlir \\
        --stop-after neura

Dependencies:
    - torch, torch-mlir (for model export)
    - mlir-neura-opt (built from this project)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _find_tool(name: str) -> str:
    """Find a built tool binary relative to this script or in PATH."""
    # Look relative to the script: tools/neura-py-frontend/neura_pipeline.py
    script_dir = Path(__file__).resolve().parent
    build_dir = script_dir.parent.parent / "build"
    candidate = build_dir / "tools" / name / name
    if candidate.exists():
        return str(candidate)
    # Fall back to PATH
    return name


# ---------------------------------------------------------------------------
# Stage 1: PyTorch → Linalg-on-Tensors (via torch-mlir)
# ---------------------------------------------------------------------------

def export_pytorch_to_linalg(
    model_file: str,
    output_mlir: str,
    func_name: str = "forward",
    example_shape: list = None,
    dynamic_dims: dict = None,
) -> bool:
    """Export a PyTorch model to Linalg-on-Tensors MLIR using torch-mlir."""
    try:
        from torch_mlir import compile as torch_mlir_compile, OutputType
    except ImportError as e:
        print(f"[neura_pipeline] torch-mlir not available: {e}", file=sys.stderr)
        print("[neura_pipeline] Please install: pip install torch-mlir", file=sys.stderr)
        return False

    import torch
    import importlib.util

    # Load the model from the given Python file
    spec = importlib.util.spec_from_file_location("user_model", model_file)
    if spec is None:
        print(f"[neura_pipeline] Cannot load model file: {model_file}", file=sys.stderr)
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The user's module should expose a `model` attribute (nn.Module instance)
    if not hasattr(module, "model"):
        print("[neura_pipeline] Model file must expose a `model` attribute "
              "(nn.Module instance)", file=sys.stderr)
        return False

    model = module.model.eval()
    shape = example_shape or [1, 3, 224, 224]
    example_input = torch.randn(*shape)

    try:
        if dynamic_dims:
            # Export with dynamic shapes using torch.export + torch_mlir.compile.
            from torch.export import Dim, export
            dynamic_shapes = {}
            for tensor_name, dim_map in dynamic_dims.items():
                dynamic_shapes[tensor_name] = {
                    dim: Dim(f"d{dim}", min=d["min"], max=d["max"])
                    for dim, d in dim_map.items()
                }
            try:
                exported = export(model, (example_input,),
                                  dynamic_shapes=dynamic_shapes)
                mlir_module = torch_mlir_compile(
                    exported, output_type=OutputType.LINALG_ON_TENSORS)
                print(f"[neura_pipeline] Exported with dynamic shapes "
                      f"→ {output_mlir}")
            except (ImportError, TypeError, AttributeError) as e:
                print(f"[neura_pipeline] Dynamic shape export failed ({e}), "
                      "falling back to static shape", file=sys.stderr)
                mlir_module = torch_mlir_compile(
                    model, example_input,
                    output_type=OutputType.LINALG_ON_TENSORS)
        else:
            mlir_module = torch_mlir_compile(
                model, example_input,
                output_type=OutputType.LINALG_ON_TENSORS,
            )
        with open(output_mlir, "w") as f:
            f.write(str(mlir_module))
        print(f"[neura_pipeline] Exported Linalg-on-Tensors IR → {output_mlir}")
        return True
    except Exception as e:
        print(f"[neura_pipeline] torch-mlir export failed: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Stage 2: Linalg-on-Tensors → Affine (via --linalg-to-affine-conversion)
# ---------------------------------------------------------------------------

def lower_linalg_to_affine(input_mlir: str, output_mlir: str) -> bool:
    """Lower Linalg-on-Tensors to Affine dialect using mlir-neura-opt."""
    tool = _find_tool("mlir-neura-opt")
    cmd = [tool, "--linalg-to-affine-conversion", input_mlir, "-o", output_mlir]
    return _run_cmd(cmd, "Linalg → Affine")


# ---------------------------------------------------------------------------
# Stage 3: Affine → Taskflow → Neura (via --taskflow-conversion)
# ---------------------------------------------------------------------------

def lower_affine_to_neura(input_mlir: str, output_mlir: str) -> bool:
    """Lower Affine to Neura dialect via Taskflow decomposition.

    The ``--taskflow-conversion`` pipeline internally performs:
        Affine → Taskflow (multi-CGRA) → Neura
    so the output is already Neura dialect IR.
    """
    tool = _find_tool("mlir-neura-opt")
    cmd = [tool, "--taskflow-conversion", input_mlir, "-o", output_mlir]
    return _run_cmd(cmd, "Affine → Taskflow → Neura")


# ---------------------------------------------------------------------------
# Stage 4: Neura → Neura optimised + Codegen (via --neura-conversion)
# ---------------------------------------------------------------------------

def lower_neura_to_codegen(
    input_mlir: str,
    output_file: str,
    arch_spec: str = None,
) -> bool:
    """Run Neura-dialect optimisation, mapping and code generation.

    The ``--neura-conversion`` pipeline performs:
        - Lower remaining mixed dialects → pure Neura
        - Canonicalize, fold constants, leverage predicated values
        - Transform control → dataflow
        - Fuse patterns, insert data movement
        - Map to accelerator, generate code (JSON instructions)
    """
    tool = _find_tool("mlir-neura-opt")
    cmd = [tool, "--neura-conversion"]
    if arch_spec:
        cmd.extend(["--architecture-spec", arch_spec])
    cmd.extend([input_mlir, "-o", output_file])
    return _run_cmd(cmd, "Neura → Optimise + Codegen")


# ---------------------------------------------------------------------------
# Full pipeline: Linalg → Affine → Taskflow → Neura → Codegen
# ---------------------------------------------------------------------------

def full_pipeline(
    input_mlir: str,
    output_file: str,
    arch_spec: str = None,
    stop_after: str = None,
    keep_intermediate: bool = False,
) -> bool:
    """Run the full Linalg-on-Tensors → Neura pipeline in three stages.

    All three stages output Neura dialect IR:

        Stage 1: Linalg-on-Tensors → Affine (via --linalg-to-affine-conversion)
        Stage 2: Affine → Neura            (via --taskflow-conversion)
        Stage 3: Neura → Optimised + Codegen (via --neura-conversion)

    Args:
        input_mlir: Path to Linalg-on-Tensors .mlir file.
        output_file: Path for the final output (.mlir or .json).
        arch_spec: Optional path to architecture specification YAML.
        stop_after: If set, stop after one of: affine, taskflow, neura.
        keep_intermediate: Keep intermediate .mlir files on disk.
    """
    tmpdir = tempfile.mkdtemp(prefix="neura_pipeline_") if not keep_intermediate else "."
    tmp = lambda name: os.path.join(tmpdir, name)

    # Stage 1: Linalg → Affine
    affine_out = tmp("01_affine.mlir")
    print("[neura_pipeline] Stage 1: Linalg → Affine ...")
    if not lower_linalg_to_affine(input_mlir, affine_out):
        _cleanup(tmpdir, keep_intermediate)
        return False
    if stop_after == "affine":
        shutil.copy(affine_out, output_file)
        print(f"[neura_pipeline] Stopped after Stage 1 (Affine). Output: {output_file}")
        _cleanup(tmpdir, keep_intermediate)
        return True

    # Stage 2: Affine → Taskflow → Neura
    neura_out = tmp("02_neura.mlir")
    print("[neura_pipeline] Stage 2: Affine → Taskflow → Neura ...")
    if not lower_affine_to_neura(affine_out, neura_out):
        _cleanup(tmpdir, keep_intermediate)
        return False
    if stop_after == "taskflow":
        shutil.copy(neura_out, output_file)
        print(f"[neura_pipeline] Stopped after Stage 2 (Neura). Output: {output_file}")
        _cleanup(tmpdir, keep_intermediate)
        return True

    # Stage 3: Neura → Optimised + Codegen
    print("[neura_pipeline] Stage 3: Neura → Optimise + Codegen ...")
    if not lower_neura_to_codegen(neura_out, output_file, arch_spec):
        _cleanup(tmpdir, keep_intermediate)
        return False
    # stop_after == "neura" (or default) — output is already at output_file

    _cleanup(tmpdir, keep_intermediate)
    return True


def _cleanup(tmpdir: str, keep: bool) -> None:
    """Remove the temporary directory unless intermediates are kept."""
    if not keep:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list, description: str) -> bool:
    """Run a command and report status."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[neura_pipeline] {description} FAILED", file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            return False
        print(f"[neura_pipeline] {description} OK")
        return True
    except FileNotFoundError:
        print(f"[neura_pipeline] Tool not found: {cmd[0]}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Python Frontend Lowering Pipeline to Neura Dialect"
    )
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--model", "-m", type=str,
        help="Path to a Python file exposing a `model` attribute (nn.Module)."
    )
    input_group.add_argument(
        "--input", "-i", type=str,
        help="Path to an existing Linalg-on-Tensors .mlir file."
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=str, default="output.json",
        help="Output file path (default: output.json)."
    )

    # Pipeline control
    parser.add_argument(
        "--stop-after", type=str, choices=["affine", "taskflow", "neura"],
        help="Stop after: affine (Stage 1), taskflow (Stage 2, output Neura), "
             "or neura (Stage 3, output optimised Neura + codegen)."
    )
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Keep intermediate .mlir files on disk."
    )

    # Model export options
    parser.add_argument(
        "--func-name", type=str, default="forward",
        help="Function name for torch-mlir export (default: forward)."
    )
    parser.add_argument(
        "--example-shape", type=int, nargs="+",
        help="Example input shape, e.g. 1 3 224 224 (default: 1 3 224 224)."
    )

    # Architecture
    parser.add_argument(
        "--arch-spec", type=str,
        help="Path to CGRA architecture specification YAML."
    )

    args = parser.parse_args()

    # Determine input MLIR file
    if args.model:
        # Stage 0: PyTorch → Linalg-on-Tensors
        linalg_mlir = args.output.replace(".json", ".mlir").replace(".yaml", ".mlir")
        if not linalg_mlir.endswith(".mlir"):
            linalg_mlir += ".mlir"
        if not export_pytorch_to_linalg(
            args.model, linalg_mlir,
            func_name=args.func_name,
            example_shape=args.example_shape,
        ):
            return 1
        input_mlir = linalg_mlir
    else:
        input_mlir = args.input
        if not os.path.exists(input_mlir):
            print(f"[neura_pipeline] Input file not found: {input_mlir}", file=sys.stderr)
            return 1

    # Run the full pipeline
    success = full_pipeline(
        input_mlir=input_mlir,
        output_file=args.output,
        arch_spec=args.arch_spec,
        stop_after=args.stop_after,
        keep_intermediate=args.keep_intermediate,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
