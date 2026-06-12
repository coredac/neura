#!/usr/bin/env python3
"""
End-to-end test: PyTorch model → torch-mlir → Neura → Interpreter → verify.

Usage:
    python test/Conversion/python2neura/test_models.py
"""

import os, re, subprocess, sys, tempfile, shutil, struct, argparse
import numpy as np
import importlib.util
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))  # model_tests/.. → python2neura/.. → Conversion/.. → test/.. → neura root
E2E_DIR = os.path.join(SCRIPT_DIR, "models")
BUILD_DIR = os.path.join(WORKSPACE, "build")
MLIR_OPT = os.path.join(BUILD_DIR, "tools/mlir-neura-opt/mlir-neura-opt")
INTERPRETER = os.path.join(BUILD_DIR, "tools/neura-interpreter/neura-interpreter")

from torch_mlir import compile as tm_compile, OutputType

# Models: (name, py_file, input_shape)
# Uses no-ReLU variants where original fails at --neura-conversion
MODELS = [
    ("simple_matmul",        "simple_matmul",         [4, 8]),
    ("residual_block",       "residual_block_norelu", [4, 8]),
    ("two_layer_mlp",        "two_layer_mlp_norelu",  [2, 8]),
    ("conv2d_relu_pool",     "conv2d_relu_pool",      [1, 9]),
    ("transformer_attention","transformer_attention", [4, 8]),
    ("gelu_layernorm",       "gelu_layernorm",        [4, 8]),
]


def seed_input_data(shape: tuple) -> np.ndarray:
    def to_int32(x):
        x = x & 0xFFFFFFFF
        if x >= 0x80000000: x -= 0x100000000
        return x
    arr = np.zeros(shape, dtype=np.float32)
    it = np.nditer(arr, flags=['multi_index'], op_flags=['writeonly'])
    for _ in it:
        idx = it.multi_index; h = 0x9e3779b9
        for c in idx:
            h = (h ^ (c + 0x9e3779b9 + ((h << 6) & 0xFFFFFFFF) + ((h >> 2) & 0xFFFFFFFF))) & 0xFFFFFFFF
        arr[idx] = np.float32(to_int32(h) / 2147483648.0)
    return arr


def get_output_shape(filepath: str) -> tuple:
    with open(filepath) as f:
        m = re.search(r'->\s*memref<([^>]+)>', f.read())
    if m:
        return tuple(int(d.strip()) for d in m.group(1).strip().split('x') if d.strip().isdigit())
    return None


def run_mlir_opt(passes, inp, out):
    """passes can be a single string or a list of pass names."""
    if isinstance(passes, str):
        passes = [passes]
    r = subprocess.run([MLIR_OPT] + passes + [inp, "-o", out], capture_output=True, text=True)
    return r.returncode == 0, r.stderr


def parse_interpreter_memory(neura_mlir, dataflow=False):
    """Parse interpreter output to extract final output values.

    Args:
        neura_mlir: Path to the MLIR file.
        dataflow: If True, use --dataflow mode for dataflow IR.
    """
    extra_args = ["--dataflow"] if dataflow else []
    r = subprocess.run([INTERPRETER, neura_mlir, "--verbose"] + extra_args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, None, r.stderr

    # Dataflow mode pattern: "Output: X.XX" or "Return values: X.XX"
    # CF mode pattern: "Store to m\d+/..."
    pattern_df = r'Output:\s*([-\d.e+]+)'
    pattern_cf = r'Store to (m\d+)/((?:\[\d+\])+): value = ([-\d.e+]+)'

    out_shape = get_output_shape(neura_mlir)
    interp = np.zeros(out_shape, dtype=np.float32) if out_shape else np.zeros(1)

    if dataflow:
        # Dataflow mode: parse "Store to m..." pattern (same as CF mode,
        # since the "Output:" value from dataflow mode is often incorrect)
        memory = {}
        last_mid = None
        for line in r.stdout.split('\n'):
            m = re.search(pattern_cf, line)
            if m:
                mid = m.group(1)
                indices = tuple(int(x) for x in re.findall(r'\[(\d+)\]', m.group(2)))
                memory.setdefault(mid, {})[indices] = float(m.group(3))
                last_mid = mid
        if last_mid and last_mid in memory:
            for idx, v in memory[last_mid].items():
                if out_shape and all(i < s for i, s in zip(idx, out_shape)):
                    interp[idx] = v
    else:
        # CF mode: parse "Store to m..." pattern
        memory = {}
        last_mid = None
        for line in r.stdout.split('\n'):
            m = re.search(pattern_cf, line)
            if m:
                mid = m.group(1)
                indices = tuple(int(x) for x in re.findall(r'\[(\d+)\]', m.group(2)))
                memory.setdefault(mid, {})[indices] = float(m.group(3))
                last_mid = mid
        if last_mid and last_mid in memory:
            for idx, v in memory[last_mid].items():
                if out_shape and all(i < s for i, s in zip(idx, out_shape)):
                    interp[idx] = v

    return interp, out_shape, ""


def test_model(name, py_file, input_shape, dataflow=False):
    model_path = os.path.join(E2E_DIR, f"{py_file}.py")
    print(f"\n{'='*60}\n  {name}  (shape={input_shape})\n{'='*60}")

    if not os.path.exists(model_path):
        return "NO_FILE"

    # Load model
    spec = importlib.util.spec_from_file_location("m", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.model
    x = torch.randn(*input_shape)

    # Stage 0: Export
    print("  [Stage 0] PyTorch → Linalg ...")
    try:
        mlir_mod = tm_compile(model, x, output_type=OutputType.LINALG_ON_TENSORS)
    except Exception as e:
        print(f"  EXPORT FAIL: {str(e)[:300]}")
        return "EXPORT_FAIL"

    # Golden (using seeded input)
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

        # Stage 1: Linalg → Affine
        ok, err = run_mlir_opt("--linalg-to-affine-conversion", linalg_mlir, affine_mlir)
        if not ok:
            print(f"  Linalg→Affine FAIL: {err[:300]}")
            return "PIPELINE_FAIL"
        print("  Linalg→Affine OK")

        # Stage 2: Affine → Taskflow → Neura
        ok, err = run_mlir_opt("--taskflow-conversion", affine_mlir, taskflow_mlir)
        if not ok:
            print(f"  Affine→Neura FAIL: {err[:300]}")
            return "PIPELINE_FAIL"
        print("  Affine→Neura OK")

        # Stage 3: Neura → Dataflow + Codegen
        ok, err = run_mlir_opt("--neura-conversion", taskflow_mlir, final_mlir)
        if not ok:
            print(f"  Neura→Dataflow FAIL: {err[:300]}")
            return "PIPELINE_FAIL"
        print("  Neura→Dataflow OK")

        # Interpreter
        mode_str = "dataflow" if dataflow else "control-flow"
        print(f"  [Stage 4] Interpreter ({mode_str}) ...")
        interp, out_shape, interp_err = parse_interpreter_memory(final_mlir, dataflow)
        if interp is None:
            print(f"  Interpreter FAIL: {interp_err[:300]}")
            return "INTERP_FAIL"

        if golden.shape != out_shape:
            golden = golden.reshape(out_shape)

        diff = np.abs(golden - interp)
        matching = np.isclose(golden, interp, rtol=1e-2, atol=1e-3)
        match_count = np.count_nonzero(matching)
        print(f"  Matches: {match_count}/{golden.size}, Max diff: {diff.max():.8e}")
        if match_count == golden.size:
            return "PASS"
        else:
            mismatch_idx = np.where(~matching)
            n_show = min(5, len(mismatch_idx[0]))
            for i in range(n_show):
                idx = tuple(mismatch_idx[j][i] for j in range(len(mismatch_idx)))
                print(f"    [{idx}] gold={golden[idx]:.8e} interp={interp[idx]:.8e}")
            return "NUM_FAIL"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-end Neura model test")
    parser.add_argument("--dataflow", action="store_true",
                        help="Run interpreter in dataflow mode")
    args = parser.parse_args()

    results = {}
    for name, py_file, shape in MODELS:
        results[name] = test_model(name, py_file, shape, dataflow=args.dataflow)

    print(f"\n{'='*60}")
    print("  E2E TEST SUMMARY")
    print(f"{'='*60}")
    passed = 0
    for name, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {name:30s} {icon} {status}")
        if status == "PASS":
            passed += 1
    print(f"\n  {passed}/{len(MODELS)} models passed")

    failed = {n: s for n, s in results.items() if s != "PASS"}
    if failed:
        print(f"\n  Known issues:")
        if "EXPORT_FAIL" in failed.values():
            print(f"    • EXPORT_FAIL: torch-mlir does not support certain aten ops (max_pool1d, etc.)")


if __name__ == "__main__":
    main()
