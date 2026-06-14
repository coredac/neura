#!/usr/bin/env python3
## @file test_models.py
## @brief End-to-end numerical correctness test for the Neura IR pipeline.
##
## @details
## For each model listed in #MODELS, this script runs:
##   1. PyTorch model -> torch-mlir Linalg IR
##   2. Linalg -> Affine -> Taskflow -> Neura -> Dataflow IR (mlir-neura-opt passes)
##   3. neura-interpreter execution on the final IR, extracting output values
##   4. PyTorch model run with the same seeded input as golden reference
##   5. Element-by-element comparison: max_abs_err, max_rel_err, mean_abs_err
##
## The input data is deterministically seeded using a C++-compatible hash so
## that both the interpreter and PyTorch golden reference see identical tensors.
##
## @usage
## @code
##   python test/Conversion/python2neura/model_tests/test_models.py
##   python test/Conversion/python2neura/model_tests/test_models.py --dataflow
## @endcode

import os
import re
import subprocess
import sys
import tempfile
import shutil
import argparse
import importlib.util

import numpy as np
import torch

from torch_mlir.fx import export_and_import, OutputType

## @name Paths
## @{

## @var SCRIPT_DIR
## @brief Absolute path to the directory containing this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

## @var WORKSPACE
## @brief Project root directory (four levels above SCRIPT_DIR).
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))

## @var E2E_DIR
## @brief Directory containing per-model .py definition files.
E2E_DIR = os.path.join(SCRIPT_DIR, "models")

## @var BUILD_DIR
## @brief Build output directory containing compiled tools.
BUILD_DIR = os.path.join(WORKSPACE, "build")

## @var MLIR_OPT
## @brief Absolute path to the mlir-neura-opt binary.
MLIR_OPT = os.path.join(BUILD_DIR, "tools/mlir-neura-opt/mlir-neura-opt")

## @var INTERPRETER
## @brief Absolute path to the neura-interpreter binary.
INTERPRETER = os.path.join(BUILD_DIR, "tools/neura-interpreter/neura-interpreter")

## @}

## @var MODELS
## @brief Model registry: (name, python_file_basename, input_shape, tolerance).
## @details Each entry corresponds to a .py file under E2E_DIR that must expose
##          a `model` attribute of type torch.nn.Module.
##          tolerance is optional; defaults to 1e-5 for max_abs_err.
MODELS = [
    ("simple_matmul",         "simple_matmul",          [4, 8]),
    ("residual_block",        "residual_block",         [4, 8]),
    ("residual_block_norelu", "residual_block_norelu",  [4, 8]),
    ("two_layer_mlp",         "two_layer_mlp",          [2, 8]),
    ("two_layer_mlp_norelu",  "two_layer_mlp_norelu",   [2, 8]),
    ("conv2d_relu_pool",      "conv2d_relu_pool",       [1, 9]),
    ("transformer_attention", "transformer_attention",  [4, 8]),
    ("transformer_block",     "transformer_block",      [4, 8], 5e-5),
    ("gelu_layernorm",        "gelu_layernorm",         [4, 8]),
]


## @brief Generate a deterministic test input using a C++-compatible hash.
## @details
## Uses a Goldfish hashing scheme over multi-dimensional index coordinates,
## producing float32 values in [-1, 1).  This ensures that both the PyTorch
## golden reference and the neura-interpreter see identical seeded input.
##
## @param shape Tensor shape as a tuple of ints.
## @returns np.ndarray of dtype float32 with values in [-1, 1).
def seed_input_data(shape: tuple) -> np.ndarray:
    def to_int32(x):
        x = x & 0xFFFFFFFF
        if x >= 0x80000000:
            x -= 0x100000000
        return x
    arr = np.zeros(shape, dtype=np.float32)
    it = np.nditer(arr, flags=['multi_index'], op_flags=['writeonly'])
    for _ in it:
        idx = it.multi_index
        h = 0x9e3779b9
        for c in idx:
            h = (h ^ (c + 0x9e3779b9 + ((h << 6) & 0xFFFFFFFF) + ((h >> 2) & 0xFFFFFFFF))) & 0xFFFFFFFF
        arr[idx] = np.float32(to_int32(h) / 2147483648.0)
    return arr


## @brief Extract the output memref shape from an MLIR function signature.
## @param filepath Path to a .mlir file.
## @returns Shape tuple of ints, or None if no memref return type is found.
def get_output_shape(filepath: str) -> tuple:
    with open(filepath) as f:
        m = re.search(r'->\s*memref<([^>]+)>', f.read())
    if m:
        return tuple(int(d.strip()) for d in m.group(1).strip().split('x') if d.strip().isdigit())
    return None


## @brief Run mlir-neura-opt with the given pass(es).
## @param passes A single pass name string or a list of pass name strings.
## @param inp Path to the input .mlir file.
## @param out Path for the output .mlir file.
## @returns Tuple of (success: bool, stderr: str).
def run_mlir_opt(passes, inp, out):
    if isinstance(passes, str):
        passes = [passes]
    r = subprocess.run([MLIR_OPT] + passes + [inp, "-o", out],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr


## @brief Run neura-interpreter on a MLIR file and parse store output.
## @details
## Parses the verbose interpreter output to extract:
##   - A reconstructed output tensor from the last-written memref
##   - The inferred output shape from the MLIR function signature
##   - A per-memref store dictionary mapping coordinates to values
##
## @param neura_mlir Path to the final MLIR file.
## @param dataflow If True, pass --dataflow to the interpreter.
## @returns Tuple of (output_tensor, output_shape, stores_dict, stderr).
##          - output_tensor: np.ndarray reconstructed from last written memref
##          - output_shape: tuple from get_output_shape()
##          - stores_dict: dict mapping mid -> {coords: value}
##          - stderr: interpreter stderr, empty string on success
def parse_interpreter_stores(neura_mlir, dataflow=False):
    extra_args = ["--dataflow"] if dataflow else []
    r = subprocess.run([INTERPRETER, neura_mlir, "--verbose"] + extra_args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, None, {}, r.stderr

    # Pattern for Store lines
    pattern = r'Store to (m\d+/(?:\[\d+\])+): value = ([-\d.e+]+)'

    stores = {}       # mid -> {coords: value}
    all_stores = []   # ordered list for last-mid logic
    last_mid = None
    for line in r.stdout.split('\n'):
        m = re.search(pattern, line)
        if m:
            mid = m.group(1).split('/')[0]
            coords_str = m.group(1).split('/', 1)[1]
            coords = tuple(int(x) for x in re.findall(r'\[(\d+)\]', coords_str))
            val = float(m.group(2))
            if mid not in stores:
                stores[mid] = {}
            stores[mid][coords] = val
            all_stores.append((mid, coords, val))
            last_mid = mid

    # Reconstruct output tensor from last written memref
    out_shape = get_output_shape(neura_mlir)
    interp = np.zeros(out_shape, dtype=np.float32) if out_shape else np.zeros(1)
    if last_mid and last_mid in stores:
        for coords, val in stores[last_mid].items():
            if out_shape and all(i < s for i, s in zip(coords, out_shape)):
                interp[coords] = val

    return interp, out_shape, stores, ""


## @brief Compare interpreter store buffers against a PyTorch reference.
## @details
## Groups stores by memref ID, infers each buffer's shape from the max
## coordinates, and for buffers whose inferred shape matches the reference
## output shape, computes max_abs_err, max_rel_err, and mean_abs_err.
##
## Among all candidates, non-zero buffers are preferred; the one with the
## lowest max_abs_err is returned.
##
## @param stores Dict mid -> {coords: value} from parse_interpreter_stores().
## @param ref_output np.ndarray of the PyTorch golden reference output.
## @param output_shape Expected output shape as a list of ints.
## @returns Dict mid -> {shape, max_abs_err, max_rel_err, mean_abs_err,
##          has_nonzero, tensor_df, tensor_ref}, filtered to the best match.
def compare_stores_to_reference(stores, ref_output, output_shape):
    results = {}
    for mid, coord_vals in stores.items():
        if not coord_vals:
            continue
        # Infer shape from max coordinates
        ndim = len(next(iter(coord_vals)))
        max_coords = [0] * ndim
        for coords in coord_vals:
            for d in range(ndim):
                if coords[d] >= max_coords[d]:
                    max_coords[d] = coords[d]
        inferred_shape = [m + 1 for m in max_coords]

        if tuple(inferred_shape) != tuple(output_shape):
            continue

        # Reconstruct tensor
        tensor = np.zeros(output_shape, dtype=np.float32)
        has_nonzero = False
        for coords, val in coord_vals.items():
            tensor[coords] = val
            if abs(val) > 1e-10:
                has_nonzero = True

        abs_err = np.abs(tensor - ref_output)
        max_abs_err = float(np.max(abs_err))
        # Relative error: avoid division by (near-)zero
        denom = np.maximum(np.abs(ref_output), 1e-10)
        rel_err = np.divide(abs_err, denom, out=np.zeros_like(abs_err),
                            where=(np.abs(ref_output) > 1e-10))
        max_rel_err = float(np.max(rel_err))
        mean_abs_err = float(np.mean(abs_err))

        results[mid] = {
            "shape": inferred_shape,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "mean_abs_err": mean_abs_err,
            "has_nonzero": has_nonzero,
            "tensor_df": tensor,
            "tensor_ref": ref_output,
        }

    if not results:
        return {}

    # Prefer non-zero buffers; then pick lowest max_abs_err
    non_zero = {k: v for k, v in results.items() if v["has_nonzero"]}
    if non_zero:
        results = non_zero
    best_key = min(results, key=lambda k: results[k]["max_abs_err"])
    return {best_key: results[best_key]}


## @brief Run the full Neura pipeline for one model and compare numerics.
## @details
## Pipeline stages:
##   - Stage 0: export PyTorch model to Linalg-on-tensors via torch-mlir
##   - Stage 1: --linalg-to-affine-conversion
##   - Stage 2: --taskflow-conversion (Affine -> Taskflow -> Neura)
##   - Stage 3: --neura-conversion (Neura -> Dataflow + Codegen)
##   - Stage 4: neura-interpreter execution
##
## A seeded input identical to the interpreter's is fed to the PyTorch model
## to produce the golden reference.  The interpreter output is then compared
## element-by-element via compare_stores_to_reference().
##
## @param name Model display name.
## @param py_file Basename of the model .py file (without .py extension).
## @param input_shape Input tensor shape as a list of ints.
## @param dataflow If True, pass --dataflow to the interpreter.
## @returns Status string: "PASS", "NO_FILE", "EXPORT_FAIL", "PIPELINE_FAIL",
##          "INTERP_FAIL", or "NUM_FAIL".
def test_model(name, py_file, input_shape, dataflow=False, tolerance=1e-5):
    model_path = os.path.join(E2E_DIR, f"{py_file}.py")
    print(f"\n{'=' * 60}\n  {name}  (shape={input_shape}  tol={tolerance:.0e})\n{'=' * 60}")

    if not os.path.exists(model_path):
        return "NO_FILE"

    # Load model
    spec = importlib.util.spec_from_file_location("m", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.model
    x = torch.randn(*input_shape)

    # Stage 0: Export
    print("  [Stage 0] PyTorch -> Linalg ...")
    try:
        mlir_mod = export_and_import(model, x, output_type=OutputType.LINALG_ON_TENSORS)
    except Exception as e:
        print(f"  EXPORT FAIL: {str(e)[:300]}")
        return "EXPORT_FAIL"

    # Golden (using seeded input, matching interpreter's hash)
    inp_data = seed_input_data(tuple(input_shape))
    golden = model(torch.tensor(inp_data)).detach().numpy()

    tmpdir = tempfile.mkdtemp(prefix=f"e2e_{name}_")
    linalg_mlir = os.path.join(tmpdir, "linalg.mlir")
    affine_mlir = os.path.join(tmpdir, "affine.mlir")
    taskflow_mlir = os.path.join(tmpdir, "taskflow.mlir")
    final_mlir = os.path.join(tmpdir, "final.mlir")

    try:
        with open(linalg_mlir, "w") as f:
            f.write(str(mlir_mod))

        # Stage 1: Linalg -> Affine
        ok, err = run_mlir_opt("--linalg-to-affine-conversion", linalg_mlir, affine_mlir)
        if not ok:
            print(f"  Linalg->Affine FAIL: {err[:300]}")
            return "PIPELINE_FAIL"
        print("  Linalg->Affine OK")

        # Stage 2: Affine -> Taskflow -> Neura
        ok, err = run_mlir_opt("--taskflow-conversion", affine_mlir, taskflow_mlir)
        if not ok:
            print(f"  Affine->Neura FAIL: {err[:300]}")
            return "PIPELINE_FAIL"
        print("  Affine->Neura OK")

        # Stage 3: Neura -> Dataflow + Codegen
        ok, err = run_mlir_opt("--neura-conversion", taskflow_mlir, final_mlir)
        if not ok:
            print(f"  Neura->Dataflow FAIL: {err[:300]}")
            return "PIPELINE_FAIL"
        print("  Neura->Dataflow OK")

        # Stage 4: Interpreter
        mode_str = "dataflow" if dataflow else "control-flow"
        print(f"  [Stage 4] Interpreter ({mode_str}) ...")
        interp, out_shape, stores, interp_err = parse_interpreter_stores(final_mlir, dataflow)
        if interp is None:
            print(f"  Interpreter FAIL: {interp_err[:300]}")
            return "INTERP_FAIL"

        if golden.shape != out_shape:
            golden = golden.reshape(out_shape)

        # Per-memref debug: show inferred shapes
        for mid in sorted(stores):
            entries = stores[mid]
            ndim_stored = len(next(iter(entries)))
            inferred = [max(c[d] for c in entries) + 1 for d in range(ndim_stored)]
            has_nz = any(abs(v) > 1e-10 for v in entries.values())
            print(f"    {mid}: shape={inferred} stores={len(entries)} non_zero={has_nz}")

        # Detailed comparison using store matching
        results = compare_stores_to_reference(stores, golden, list(golden.shape))

        if not results:
            # Fallback: simple diff against last-memref reconstruction
            diff = np.abs(golden - interp)
            matching = np.isclose(golden, interp, rtol=1e-2, atol=1e-3)
            match_count = np.count_nonzero(matching)
            print(f"  Matches: {match_count}/{golden.size}, Max diff: {diff.max():.8e}")
            if match_count == golden.size:
                return "PASS"
            mismatch_idx = np.where(~matching)
            n_show = min(5, len(mismatch_idx[0]))
            for i in range(n_show):
                idx = tuple(mismatch_idx[j][i] for j in range(len(mismatch_idx)))
                print(f"    [{idx}] gold={golden[idx]:.8e} interp={interp[idx]:.8e}")
            return "NUM_FAIL"

        for mid, r in results.items():
            err_pass = r["max_abs_err"] < tolerance
            icon = "PASS" if err_pass else "FAIL"
            has_nz = "non-zero" if r["has_nonzero"] else "ALL ZERO"
            print(f"  {icon} {mid} shape={r['shape']} | "
                  f"max_abs_err={r['max_abs_err']:.2e} "
                  f"max_rel_err={r['max_rel_err']:.2e} "
                  f"mean_abs_err={r['mean_abs_err']:.2e} "
                  f"({has_nz} tol={tolerance:.0e})")
            if not err_pass:
                df_vals = r['tensor_df'].flatten()[:5]
                ref_vals = r['tensor_ref'].flatten()[:5]
                print(f"       DF[0:5]:  {df_vals}")
                print(f"       REF[0:5]: {ref_vals}")

        return "PASS" if results and all(
            r["max_abs_err"] < tolerance for r in results.values()) else "NUM_FAIL"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


## @brief Main entry point for the end-to-end test runner.
## @details Parses --dataflow flag, iterates over all models in #MODELS,
##          and prints a summary table with pass/fail counts.
def main():
    parser = argparse.ArgumentParser(description="End-to-end Neura model test")
    parser.add_argument("--dataflow", action="store_true",
                        help="Run interpreter in dataflow mode")
    args = parser.parse_args()

    results = {}
    for entry in MODELS:
        name, py_file, shape = entry[:3]
        tolerance = entry[3] if len(entry) > 3 else 1e-5
        results[name] = test_model(name, py_file, shape, dataflow=args.dataflow, tolerance=tolerance)

    print(f"\n{'=' * 60}")
    print("  E2E TEST SUMMARY")
    print(f"{'=' * 60}")
    passed = 0
    for name, status in results.items():
        icon = "OK" if status == "PASS" else "FAIL"
        print(f"  {name:30s} {icon} {status}")
        if status == "PASS":
            passed += 1
    print(f"\n  {passed}/{len(MODELS)} models passed")

    failed = {n: s for n, s in results.items() if s != "PASS"}
    if failed:
        print(f"\n  Known issues:")
        if "EXPORT_FAIL" in failed.values():
            print(f"    - EXPORT_FAIL: torch-mlir does not support certain aten ops (max_pool1d, etc.)")


if __name__ == "__main__":
    main()
