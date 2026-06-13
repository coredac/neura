#!/usr/bin/env python3
"""
Compare DF interpreter numerical results against PyTorch reference computation.

For each model:
  1. Convert Neura IR → DF IR (using DATAFLOW_PASSES)
  2. Run neura-interpreter --dataflow, extract store values from output
  3. Run PyTorch model with same input/weight data, get reference output
  4. Compare element-by-element: max absolute error, max relative error

The C++ interpreter initializes:
  - memref inputs from memref.get_global → uses global constant data
  - other memref inputs → deterministic hash from coordinates
  - intermediate/output buffers → hash-based

Both C++ and Python use the same hash algorithm.
"""

import os
import re
import subprocess
import sys
import importlib
import struct
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "build")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
GENERATED_DIR = os.path.join(SCRIPT_DIR, "generated")

OPT = os.path.join(BUILD_DIR, "tools", "mlir-neura-opt", "mlir-neura-opt")
INTERP = os.path.join(BUILD_DIR, "tools", "neura-interpreter", "neura-interpreter")

DATAFLOW_PASSES = [
    "--assign-accelerator",
    "--lower-arith-to-neura",
    "--lower-memref-to-neura",
    "--lower-builtin-to-neura",   # skip, may not exist
    "--lower-llvm-to-neura",      # skip, may not exist
    "--canonicalize-return",
    "--canonicalize-cast",
    "--promote-input-arg-to-const",
    "--fold-constant",
    "--canonicalize-live-in",
    "--leverage-predicated-value",
    "--transform-ctrl-to-data-flow",
    "--fold-constant",
    "--fuse-pattern",
    "--insert-data-mov",
]

MODELS = [
    ("simple_matmul",        "simple_matmul",         [4, 8]),
    ("residual_block",       "residual_block",        [4, 8]),
    ("residual_block_norelu","residual_block_norelu", [4, 8]),
    ("two_layer_mlp",        "two_layer_mlp",         [2, 8]),
    ("two_layer_mlp_norelu", "two_layer_mlp_norelu",  [2, 8]),
    ("conv2d_relu_pool",     "conv2d_relu_pool",      [1, 9]),
    ("transformer_attention","transformer_attention", [4, 8]),
    ("gelu_layernorm",       "gelu_layernorm",        [4, 8]),
]


def cpp_hash_value(coords):
    """Replicate the C++ hash: deterministic float in [-1, 1) from coordinates."""
    hash_val = 0x9e3779b9
    for c in coords:
        hash_val ^= (c & 0xFFFFFFFFFFFFFFFF) + 0x9e3779b9 + (hash_val << 6) + (hash_val >> 2)
        hash_val &= 0xFFFFFFFFFFFFFFFF
    val = (hash_val & 0xFFFFFFFF)
    # Sign-extend to int32
    if val & 0x80000000:
        val = val - 0x100000000
    return float(val) / 2147483648.0


def generate_input_tensor(shape):
    """Generate input tensor using same hash as C++ interpreter."""
    data = np.zeros(shape, dtype=np.float32)
    indices = list(np.ndindex(*shape))
    for idx_flat in range(len(indices)):
        idx = indices[idx_flat]
        data[idx] = cpp_hash_value(list(idx))
    return data


def run_df_pipeline(neura_mlir, df_mlir):
    """Convert Neura IR → DF IR."""
    if not os.path.exists(neura_mlir):
        return False
    result = subprocess.run(
        [OPT] + DATAFLOW_PASSES + [neura_mlir, "-o", df_mlir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    [DF pipeline] FAILED:\n{result.stderr[:500]}", file=sys.stderr)
        return False
    return True


def run_df_interpreter(df_mlir):
    """Run neura-interpreter on DF IR, capture stdout."""
    if not os.path.exists(df_mlir):
        return None
    try:
        result = subprocess.run(
            [INTERP, df_mlir, "--dataflow"],
            capture_output=True, text=True, timeout=300,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"    [DF interpreter] TIMEOUT", file=sys.stderr)
        return None
    except Exception as e:
        print(f"    [DF interpreter] ERROR: {e}", file=sys.stderr)
        return None


def parse_stores(output):
    """Parse interpreter output: extract all Store lines → dict {mem_key: value}."""
    stores = {}
    for line in output.splitlines():
        m = re.search(r"Store to (m\d+/(?:\[\d+\])+): value = ([-.\de+]+|nan|inf|-inf)", line)
        if m:
            key = m.group(1)
            try:
                val = float(m.group(2))
                stores[key] = val
            except ValueError:
                continue
    # Also extract Output line
    output_val = None
    for line in output.splitlines():
        m = re.search(r"→\s*Output:\s*([-.\de+]+)", line)
        if m:
            try:
                output_val = float(m.group(1))
            except ValueError:
                pass
            break
    return stores, output_val


def run_pytorch_reference(py_file, input_shape):
    """Import PyTorch model and run forward with hash-generated input.

    Returns:
        output_numpy: np.ndarray of model output
        output_shape: list of ints
    """
    import torch
    import importlib.util

    # Import the model module fresh to avoid cached state
    spec = importlib.util.spec_from_file_location(
        f"_ref_{py_file}", os.path.join(MODELS_DIR, f"{py_file}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.model  # Module-level `model = Model().eval()`
    input_tensor = torch.from_numpy(generate_input_tensor(input_shape))
    with torch.no_grad():
        output = model(input_tensor)
    return output.numpy(), list(output.shape)


def compare_stores_to_reference(stores, ref_output, output_shape, model_name):
    """Compare DF interpreter store values to PyTorch reference output tensor.

    The interpreter uses "mN/[i][j]/..." keys. The output memref is typically
    the last allocated buffer. We reconstruct the expected tensor and compare.
    """
    # Determine which memref ID is the output
    # The output buffer is typically allocated with memref.alloc (not from get_global)
    # In the MLIR: %alloc = memref.alloc() -> becomes output memref
    # The C++ interpreter assigns memref IDs based on order of appearance.
    # Let's find the memref IDs that have stored values matching output_shape

    # Group stores by memref ID
    memref_stores = {}
    for key, val in stores.items():
        # key format: "mN/[i][j]/..."
        m = re.match(r"(m\d+)/((?:\[\d+\])+)", key)
        if m:
            mid = m.group(1)
            coords_str = m.group(2)
            coords = [int(c) for c in re.findall(r"\[(\d+)\]", coords_str)]
            if mid not in memref_stores:
                memref_stores[mid] = {}
            memref_stores[mid][tuple(coords)] = val

    # For each memref, check if shape matches output_shape
    results = {}
    for mid, coord_vals in memref_stores.items():
        if not coord_vals:
            continue
        # Infer shape from max coords
        max_coords = [0] * len(list(coord_vals.keys())[0])
        for coords in coord_vals:
            for d, c in enumerate(coords):
                if c >= max_coords[d]:
                    max_coords[d] = c
        inferred_shape = [m + 1 for m in max_coords]

        if tuple(inferred_shape) == tuple(output_shape):
            # Reconstruct tensor
            tensor = np.zeros(output_shape, dtype=np.float32)
            has_nonzero = False
            for coords, val in coord_vals.items():
                tensor[coords] = val
                if abs(val) > 1e-10:
                    has_nonzero = True

            # Compute error metrics
            abs_err = np.abs(tensor - ref_output)
            max_abs_err = np.max(abs_err)
            rel_err = np.where(np.abs(ref_output) > 1e-10,
                               abs_err / (np.abs(ref_output) + 1e-10), 0)
            max_rel_err = np.max(rel_err)
            mean_abs_err = np.mean(abs_err)

            results[mid] = {
                "shape": inferred_shape,
                "max_abs_err": max_abs_err,
                "max_rel_err": max_rel_err,
                "mean_abs_err": mean_abs_err,
                "has_nonzero": has_nonzero,
                "tensor_df": tensor,
                "tensor_ref": ref_output,
            }

    return results


def select_best_match(results):
    """From multiple candidate memrefs with the same output shape, select
    the best one (prefer non-zero values, then lowest error)."""
    if not results:
        return results
    if len(results) == 1:
        return results

    # Prefer results with non-zero values
    non_zero = {k: v for k, v in results.items() if v["has_nonzero"]}
    if non_zero:
        results = non_zero

    # Pick the one with lowest max_abs_err
    best_key = min(results, key=lambda k: results[k]["max_abs_err"])
    return {best_key: results[best_key]}


def main():
    print("=" * 80)
    print("  DF Numerical Correctness Comparison: DF Interpreter vs PyTorch Reference")
    print("=" * 80)

    all_pass = True
    summary = []

    for name, py_file, shape in MODELS:
        print(f"\n{'─' * 60}")
        print(f"  Model: {name}  (input_shape={shape})")
        print(f"{'─' * 60}")

        neura_mlir = os.path.join(GENERATED_DIR, f"{name}_neura.mlir")
        if not os.path.exists(neura_mlir):
            print(f"  SKIP: {neura_mlir} not found (run generate_mlir.py first)")
            continue

        # Step 1: Convert to DF IR
        df_mlir = f"/tmp/{name}_df.mlir"
        if not run_df_pipeline(neura_mlir, df_mlir):
            summary.append((name, "DF_CONV_FAIL", None, None))
            all_pass = False
            continue

        # Step 2: Run DF interpreter
        interp_out = run_df_interpreter(df_mlir)
        if interp_out is None:
            summary.append((name, "DF_INTERP_FAIL", None, None))
            all_pass = False
            continue

        stores, output_val = parse_stores(interp_out)
        total_stores = len(stores)
        print(f"  DF stores captured: {total_stores}")

        # Debug: show memref IDs and inferred shapes
        memref_entries = {}
        for k, v in stores.items():
            mid = k.split('/')[0]
            # Parse index part: e.g., "m0/[0][1]" → tail = "[0][1]"
            tail = k[len(mid) + 1:]  # everything after "m0/"
            coord = tuple(int(x) for x in re.findall(r'\[(\d+)\]', tail))
            if mid not in memref_entries:
                memref_entries[mid] = []
            memref_entries[mid].append((coord, v))
        for mid, entries in sorted(memref_entries.items()):
            ndim = len(entries[0][0])
            inferred = [max(c[i] for c, _ in entries) + 1 for i in range(ndim)]
            has_nz = any(abs(v) > 1e-10 for _, v in entries)
            print(f"    {mid}: shape={inferred} stores={len(entries)} non_zero={has_nz}")

        # Step 3: Run PyTorch reference
        try:
            ref_output, out_shape = run_pytorch_reference(py_file, shape)
            print(f"  PyTorch reference output shape: {out_shape}")
        except Exception as e:
            print(f"  PyTorch REFERENCE ERROR: {e}", file=sys.stderr)
            summary.append((name, "PYTORCH_FAIL", None, None))
            all_pass = False
            continue

        # Step 4: Compare
        results = compare_stores_to_reference(stores, ref_output, out_shape, name)
        results = select_best_match(results)

        if not results:
            print(f"  ⚠️  No matching output memref found in DF stores")
            print(f"     Output value from return: {output_val}")
            print(f"     First few ref values: {ref_output.flatten()[:5]}")
            summary.append((name, "NO_MATCH", None, None))
            all_pass = False
        else:
            for mid, r in results.items():
                status = "✅" if r["max_abs_err"] < 2e-2 else "❌"
                has_nz = "non-zero" if r["has_nonzero"] else "ALL ZERO"
                print(f"  {status} {mid} shape={r['shape']} | "
                      f"max_abs_err={r['max_abs_err']:.2e} "
                      f"max_rel_err={r['max_rel_err']:.2e} "
                      f"mean_abs_err={r['mean_abs_err']:.2e} "
                      f"({has_nz})")
                if r["max_abs_err"] > 1e-5:
                        df_vals = r['tensor_df'].flatten()[:5]
                        ref_vals = r['tensor_ref'].flatten()[:5]
                        print(f"       DF[0:5]:  {df_vals}")
                        print(f"       REF[0:5]: {ref_vals}")
                all_pass = all_pass and (r["max_abs_err"] < 2e-2)
                summary.append((name, "OK" if r["max_abs_err"] < 2e-2 else "MISMATCH",
                                r["max_abs_err"], r["max_rel_err"]))

        # Cleanup
        if os.path.exists(df_mlir):
            os.remove(df_mlir)

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    for item in summary:
        name, status, max_ae, max_re = item
        icon = "✅" if status == "OK" else "❌"
        if max_ae is not None:
            print(f"  {icon} {name:30s}  {status:10s}  max_abs_err={max_ae:.2e}  max_rel_err={max_re:.2e}")
        else:
            print(f"  {icon} {name:30s}  {status}")

    if all_pass:
        print(f"\n  🎉 ALL MODELS PASS numerical comparison!")
    else:
        print(f"\n  ⚠️  SOME MODELS HAVE NUMERICAL DISCREPANCIES")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
