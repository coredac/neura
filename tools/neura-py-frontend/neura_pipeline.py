#!/usr/bin/env python3
# @file neura_pipeline.py
# @brief Python Frontend Lowering Pipeline to Neura Dialect
# @details
# This script provides the complete lowering path from PyTorch models to the
# Neura CGRA dataflow dialect, executed in three stages:
# @verbatim
#     PyTorch model
#       → torch-mlir (Linalg-on-Tensors IR)
#         → Stage 1: --linalg-to-affine-conversion  (→ Affine)
#         → Stage 2: --taskflow-conversion           (→ Neura)
#         → Stage 3: --neura-conversion              (→ Neura optimised + Codegen)
#           → CGRA JSON instructions
# @endverbatim
# All three stages run through mlir-neura-opt. The final output is fully
# lowered to the Neura dialect with code generation.
#
# Usage:
# @code
#     # Generate MLIR from a PyTorch model and lower through the full pipeline
#     python neura_pipeline.py --model model.py --output out.json
#
#     # Or pass an existing Linalg-on-Tensors .mlir file through the pipeline
#     python neura_pipeline.py --input linalg_ir.mlir --output out.json
#
#     # Stop after Stage 2 — output is Neura dialect before codegen
#     python neura_pipeline.py --input linalg_ir.mlir --output out.mlir \\
#         --stop-after taskflow
# @endcode
# Dependencies:
#     - torch, torch-mlir (for model export)
#     - mlir-neura-opt (built from this project)

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# @name Path resolution

## @brief Find a built tool binary relative to this script or in PATH.
## @param name Tool name to find.
## @return Absolute path to the tool binary, or @p name if not found locally.
def _find_tool(name: str) -> str:
    # Look relative to the script: tools/neura-py-frontend/neura_pipeline.py
    script_dir = Path(__file__).resolve().parent
    build_dir = script_dir.parent.parent / "build"
    candidate = build_dir / "tools" / name / name
    if candidate.exists():
        return str(candidate)
    # Fall back to PATH
    return name


# @name Stage 1: PyTorch → Linalg-on-Tensors (via torch-mlir)

## @brief Export a PyTorch model to Linalg-on-Tensors MLIR using torch-mlir.
## @param model_file Path to Python file exposing a `model` attribute.
## @param output_mlir Path to write the output MLIR.
## @param func_name Function name for export (default: "forward").
## @param example_shape Example input tensor shape as a list of ints.
## @param dynamic_dims Optional dict mapping tensor names to dimension specs.
## @returns @c True on success, @c False on failure.
def export_pytorch_to_linalg(
    model_file: str,
    output_mlir: str,
    func_name: str = "forward",
    example_shape: list = None,
    dynamic_dims: dict = None,
) -> bool:
    try:
        from torch_mlir.fx import export_and_import
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
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)

    # The user's module should expose a `model` attribute (nn.Module instance)
    if not hasattr(user_module, "model"):
        print("[neura_pipeline] Model file must expose a `model` attribute "
              "(nn.Module instance)", file=sys.stderr)
        return False

    model = user_module.model.eval()
    shape = example_shape or [1, 3, 224, 224]
    example_input = torch.randn(*shape)

    try:
        if dynamic_dims:
            # Build torch.export dynamic_shapes from user-supplied dim specs.
            from torch.export import Dim
            from torch.utils._pytree import tree_flatten
            dynamic_shapes = {}
            flat_args, _ = tree_flatten((example_input,))
            input_name = "input"
            dim_map = {}
            for dim_idx, d in dynamic_dims.get(input_name, {}).items():
                dim_map[int(dim_idx)] = Dim(f"d{dim_idx}", min=d.get("min", 1))
            if dim_map:
                dynamic_shapes[input_name] = dim_map
            mlir_module = export_and_import(
                model, example_input,
                output_type="linalg-on-tensors",
                func_name=func_name,
                dynamic_shapes=dynamic_shapes or None,
            )
            print(f"[neura_pipeline] Exported with dynamic shapes "
                  f"→ {output_mlir}")
        else:
            mlir_module = export_and_import(
                model, example_input,
                output_type="linalg-on-tensors",
                func_name=func_name,
            )
        with open(output_mlir, "w") as f:
            f.write(str(mlir_module.operation))
        print(f"[neura_pipeline] Exported Linalg-on-Tensors IR → {output_mlir}")
        return True
    except Exception as e:
        print(f"[neura_pipeline] torch-mlir export failed: {e}", file=sys.stderr)
        return False


# @name Stage 2: Linalg-on-Tensors → Affine (via --linalg-to-affine-conversion)

## @brief Lower Linalg-on-Tensors to Affine dialect using mlir-neura-opt.
## @param input_mlir Path to Linalg-on-Tensors .mlir file.
## @param output_mlir Path to write the Affine dialect output.
## @returns @c True on success, @c False on failure.
def lower_linalg_to_affine(input_mlir: str, output_mlir: str) -> bool:
    tool = _find_tool("mlir-neura-opt")
    cmd = [tool, "--linalg-to-affine-conversion", input_mlir, "-o", output_mlir]
    return _run_cmd(cmd, "Linalg → Affine")


# @name Stage 3: Affine → Taskflow → Neura (via --taskflow-conversion)

## @brief Lower Affine to Neura dialect via Taskflow decomposition.
## @details
## The @c --taskflow-conversion pipeline internally performs:
## @verbatim
##     Affine → Taskflow (multi-CGRA) → Neura
## @endverbatim
## so the output is already Neura dialect IR.
## @param input_mlir Path to Affine .mlir file.
## @param output_mlir Path to write the Neura dialect output.
## @returns @c True on success, @c False on failure.
def lower_affine_to_neura(input_mlir: str, output_mlir: str) -> bool:
    tool = _find_tool("mlir-neura-opt")
    cmd = [tool, "--taskflow-conversion", input_mlir, "-o", output_mlir]
    return _run_cmd(cmd, "Affine → Taskflow → Neura")


# @name Stage 4: Neura → Neura optimised + Codegen (via --neura-conversion)

## @brief Run Neura-dialect optimisation, mapping and code generation.
## @details
## The @c --neura-conversion pipeline performs:
##     - Lower remaining mixed dialects → pure Neura
##     - Canonicalize, fold constants, leverage predicated values
##     - Transform control → dataflow
##     - Fuse patterns, insert data movement
##     - Map to accelerator, generate code (JSON instructions)
## @param input_mlir Path to Neura .mlir file.
## @param output_file Path for the final output (.mlir or .json).
## @param arch_spec Optional path to architecture specification YAML.
## @returns @c True on success, @c False on failure.
def lower_neura_to_codegen(
    input_mlir: str,
    output_file: str,
    arch_spec: str = None,
) -> bool:
    tool = _find_tool("mlir-neura-opt")
    cmd = [tool, "--neura-conversion"]
    if arch_spec:
        cmd.extend(["--architecture-spec", arch_spec])
    cmd.extend([input_mlir, "-o", output_file])
    return _run_cmd(cmd, "Neura → Optimise + Codegen")


# @name Full pipeline: Linalg → Affine → Taskflow → Neura → Codegen

## @brief Run the full Linalg-on-Tensors → Neura pipeline in three stages.
## @details
## All three stages output Neura dialect IR:
##     Stage 1: Linalg-on-Tensors → Affine (via @c --linalg-to-affine-conversion)
##     Stage 2: Affine → Neura            (via @c --taskflow-conversion)
##     Stage 3: Neura → Optimised + Codegen (via @c --neura-conversion)
## @param input_mlir Path to Linalg-on-Tensors .mlir file.
## @param output_file Path for the final output (.mlir or .json).
## @param arch_spec Optional path to architecture specification YAML.
## @param stop_after If set, stop after one of: affine, taskflow, neura.
## @param keep_intermediate Keep intermediate .mlir files on disk.
## @returns @c True on success, @c False on failure.
def full_pipeline(
    input_mlir: str,
    output_file: str,
    arch_spec: str = None,
    stop_after: str = None,
    keep_intermediate: bool = False,
) -> bool:
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


## @brief Remove the temporary directory unless intermediates are kept.
## @param tmpdir Path to the temporary directory.
## @param keep If @c True, the directory is preserved.
def _cleanup(tmpdir: str, keep: bool) -> None:
    if not keep:
        shutil.rmtree(tmpdir, ignore_errors=True)


# @name Helpers

## @brief Run a command and report status.
## @param cmd List of command-line arguments.
## @param description Human-readable description of the step.
## @returns @c True if the command succeeds, @c False otherwise.
def _run_cmd(cmd: list, description: str) -> bool:
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


# @name CLI entry point

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
