#!/usr/bin/env python3
"""
Generate MLIR for all models in models/ using neura_pipeline.py,
and optionally produce lit-testable .mlir files.

Output per model:
    <name>_neura.mlir     — Neura dialect with lit header (dataflow + interpreter)

Each generated .mlir file is a self-contained lit test that verifies:
  1. Dataflow conversion: Neura IR → dataflow IR (structural check)
  2. Interpreter: neura-interpreter executes the dataflow IR and produces
     correct values (value check)

Lit mode (--lit) additionally produces:
    lit/<name>.mlir       — Copy of <name>_neura.mlir in a separate directory

Usage:
    python generate_mlir.py [--output-dir generated] [--lit]
"""

import os
import re
import subprocess
import sys
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add neura-py-frontend to sys.path so we can import neura_pipeline
FRONTEND_DIR = os.path.abspath(os.path.join(
    SCRIPT_DIR, "..", "..", "..", "..", "tools", "neura-py-frontend"
))
sys.path.insert(0, FRONTEND_DIR)

from neura_pipeline import (
    export_pytorch_to_linalg,
    lower_linalg_to_affine,
    lower_affine_to_neura,
    lower_neura_to_codegen,
)

MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated")
LIT_DIR = os.path.join(OUTPUT_DIR, "lit")

# Models to generate: (name, python_file, example_shape)
# Each .py file must expose a `model` attribute (nn.Module instance)
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

# ---------------------------------------------------------------------------
# Dataflow conversion pass pipeline
# (same as --neura-conversion but stops before --map-to-accelerator
#  and --generate-code, giving pure dataflow IR for checking)
# ---------------------------------------------------------------------------
DATAFLOW_PASSES = [
    "--assign-accelerator",
    "--lower-affine-to-neura",
    "--lower-arith-to-neura",
    "--lower-memref-to-neura",
    "--lower-builtin-to-neura",
    "--lower-llvm-to-neura",
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


def _find_opt() -> str:
    """Find the mlir-neura-opt binary."""
    build_dir = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "build")
    candidate = os.path.join(build_dir, "tools", "mlir-neura-opt", "mlir-neura-opt")
    if os.path.exists(candidate):
        return candidate
    return "mlir-neura-opt"


def _find_interpreter() -> str:
    """Find the neura-interpreter binary."""
    build_dir = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "build")
    candidate = os.path.join(
        build_dir, "tools", "neura-interpreter", "neura-interpreter"
    )
    if os.path.exists(candidate):
        return candidate
    return "neura-interpreter"


def _run_dataflow_passes(neura_mlir: str, output_mlir: str) -> bool:
    """Run dataflow conversion passes on Neura IR.

    Args:
        neura_mlir: Path to Neura IR (clean, no lit header).
        output_mlir: Path to write the dataflow IR output.

    Returns:
        True on success, False on failure.
    """
    opt = _find_opt()
    result = subprocess.run(
        [opt] + DATAFLOW_PASSES + [neura_mlir, "-o", output_mlir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [dataflow] FAILED:\n{result.stderr}", file=sys.stderr)
        return False
    return True


def _run_dataflow_interpreter(dataflow_mlir: str):
    """Run neura-interpreter on the dataflow MLIR.

    Args:
        dataflow_mlir: Path to dataflow IR .mlir file.

    Returns:
        stdout string on success, None on failure.
    """
    interp = _find_interpreter()
    result = subprocess.run(
        [interp, dataflow_mlir, "--verbose", "--dataflow"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [interpreter] FAILED (rc={result.returncode}):\n"
              f"{result.stderr[:500]}", file=sys.stderr)
        return None
    return result.stdout


def _generate_dataflow_checks(dataflow_mlir: str, interpreter_output: str) -> str:
    """Generate FileCheck patterns from dataflow IR and interpreter output.

    Args:
        dataflow_mlir: Path to the dataflow IR .mlir file.
        interpreter_output: stdout from neura-interpreter --verbose --dataflow.

    Returns:
        A string of FileCheck check lines with DATAFLOW_IR and
        INTERPRETER_OUTPUT prefixes.
    """
    checks = []

    # Read the dataflow IR content
    with open(dataflow_mlir) as f:
        df_content = f.read()

    # --- DATAFLOW_IR: check key structural elements using DAG ---
    # Use CHECK-DAG so that FileCheck does not care about the order.
    # This avoids issues with complex MLIR syntax and regex metacharacters.

    # Module must have torch.debug_module_name
    checks.append('// DATAFLOW_IR-DAG: module attributes {torch.debug_module_name')

    # Function-level: must carry dataflow_mode attribute
    checks.append('// DATAFLOW_IR-DAG: func.func @forward')
    checks.append('// DATAFLOW_IR-DAG: dataflow_mode = "predicate"')

    # Kernel ops
    checks.append("// DATAFLOW_IR-DAG: neura.kernel")

    # Key dataflow ops that prove conversion happened
    checks.append("// DATAFLOW_IR-DAG: neura.counter")
    checks.append("// DATAFLOW_IR-DAG: neura.load_indexed")
    checks.append("// DATAFLOW_IR-DAG: neura.store_indexed")

    # Compute ops (capture whichever appear)
    _ALL_COMPUTE_OPS = [
        "neura.fmul_fadd",
        "neura.fadd",
        "neura.relu",
        "neura.tanh",
        "neura.sigmoid",
        "neura.erf",
        "neura.fcmp",
        "neura.fsub",
        "neura.fdiv",
        "neura.fmax",
        "neura.fmul",
    ]
    present_ops = [op for op in _ALL_COMPUTE_OPS if op in df_content]
    selected = []
    for op in present_ops:
        if any(longer for longer in present_ops if longer != op and op in longer):
            continue
        selected.append(op)
    for op_name in selected:
        checks.append(f"// DATAFLOW_IR-DAG: {op_name}")

    # Yield
    checks.append("// DATAFLOW_IR-DAG: neura.yield")

    # --- INTERPRETER_OUTPUT: check final output value and no errors ---
    # Dataflow execution order of concurrent stores is non-deterministic,
    # so we only check the final Output value (which is stable) and store
    # existence.
    if interpreter_output:
        # Store existence check (structural — at least one store happened)
        if re.search(r'Store to m\d+/', interpreter_output):
            checks.append("// INTERPRETER_OUTPUT-DAG: Store to m")

        # Final Output line with concrete value (deterministic)
        for line in interpreter_output.split("\n"):
            m = re.search(r'Output:\s*([-\d.e+]+)', line)
            if m:
                checks.append(f"// INTERPRETER_OUTPUT-DAG: Output: {m.group(1)}")
                break

        # No errors during execution
        checks.append("// INTERPRETER_OUTPUT-NOT: Error")
        checks.append("// INTERPRETER_OUTPUT-NOT: Failed")
        checks.append("// INTERPRETER_OUTPUT-NOT: Unhandled")

    return "\n".join(checks) + "\n"


def _lit_header() -> str:
    """Return the lit RUN lines for a dataflow + interpreter test.

    RUN lines go at the TOP of the file.
    CHECK lines go at the BOTTOM (see _lit_footer).

    The test pipeline:
      1. Convert Neura IR → dataflow IR  (using DATAFLOW_PASSES)
      2. Run neura-interpreter on dataflow IR
      3. Check dataflow IR structure      (DATAFLOW_IR FileCheck)
      4. Check interpreter output values  (INTERPRETER_OUTPUT FileCheck)
    """
    passes_str = " ".join(DATAFLOW_PASSES)
    return (
        f"// RUN: mlir-neura-opt {passes_str} %s -o %t_dataflow.mlir\n"
        f"// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow"
        f" > %t_output.txt\n"
        f"// RUN: FileCheck %s --check-prefix=DATAFLOW_IR"
        f" --input-file=%t_dataflow.mlir\n"
        f"// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT"
        f" --input-file=%t_output.txt\n"
    )


def _lit_footer(dataflow_checks: str) -> str:
    """Return the CHECK lines to append at the END of the file."""
    return "\n" + dataflow_checks


def _dataflow_fallback_checks() -> str:
    """Minimal checks when dataflow conversion fails."""
    return (
        "// DATAFLOW_IR: neura.kernel\n"
        "// INTERPRETER_OUTPUT-NOT: Error\n"
    )


def _assemble_lit_file(neura_mlir: str, dataflow_checks: str) -> None:
    """Rewrite the .mlir file with RUN lines at top, CHECK lines at bottom."""
    with open(neura_mlir) as f:
        content = f.read()
    with open(neura_mlir, "w") as f:
        f.write(_lit_header())
        f.write(content)
        f.write(_lit_footer(dataflow_checks))


def _generate_lit_test(name: str, neura_mlir: str, lit_dir: str) -> str:
    """Copy the headered Neura .mlir file into the lit directory."""
    lit_mlir = os.path.join(lit_dir, f"{name}.mlir")
    shutil.copy(neura_mlir, lit_mlir)
    return lit_mlir


def _strip_taskflow(neura_mlir: str) -> bool:
    """Strip taskflow.task wrappers from a Neura .mlir file in-place."""
    opt = _find_opt()
    tmp = neura_mlir + ".tmp"
    result = subprocess.run(
        [opt, "--strip-taskflow-task", neura_mlir, "-o", tmp],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [strip-taskflow-task] FAILED:\n{result.stderr}", file=sys.stderr)
        return False
    os.replace(tmp, neura_mlir)
    return True


def generate_model(name: str, py_file: str, example_shape: list,
                   lit_dir: str = None) -> bool:
    print(f"\n{'='*60}")
    print(f"  {name}  (shape={example_shape})")
    print(f"{'='*60}")

    model_path = os.path.join(MODELS_DIR, f"{py_file}.py")
    if not os.path.exists(model_path):
        print(f"  SKIP: model file not found: {model_path}")
        return False

    # Stage 0: PyTorch → Linalg
    linalg_mlir = os.path.join(OUTPUT_DIR, f"{name}_linalg.mlir")
    print(f"  [Stage 0] PyTorch → Linalg ...")
    if not export_pytorch_to_linalg(model_path, linalg_mlir, example_shape=example_shape):
        print(f"  FAIL at Stage 0")
        return False

    # Stage 1: Linalg → Affine
    affine_mlir = os.path.join(OUTPUT_DIR, f"{name}_affine.mlir")
    print(f"  [Stage 1] Linalg → Affine ...")
    if not lower_linalg_to_affine(linalg_mlir, affine_mlir):
        print(f"  FAIL at Stage 1")
        return False

    # Stage 2: Affine → Neura
    neura_mlir = os.path.join(OUTPUT_DIR, f"{name}_neura.mlir")
    print(f"  [Stage 2] Affine → Taskflow → Neura ...")
    if not lower_affine_to_neura(affine_mlir, neura_mlir):
        print(f"  FAIL at Stage 2")
        return False

    # Strip taskflow.task wrappers, producing clean Neura IR
    print(f"  [Strip] Removing taskflow.task wrappers ...")
    if not _strip_taskflow(neura_mlir):
        print(f"  FAIL at strip-taskflow-task")
        return False

    # --- Generate dataflow IR + interpreter output for CHECK lines ---
    df_mlir = os.path.join(OUTPUT_DIR, f"{name}_dataflow.mlir")
    print(f"  [Dataflow] Converting Neura → Dataflow ...")
    if _run_dataflow_passes(neura_mlir, df_mlir):
        print(f"  [Interpreter] Running on dataflow IR ...")
        interp_output = _run_dataflow_interpreter(df_mlir)
        dataflow_checks = _generate_dataflow_checks(df_mlir, interp_output or "")
        print(f"  [Checks] Generated DATAFLOW_IR + INTERPRETER_OUTPUT checks")
    else:
        dataflow_checks = _dataflow_fallback_checks()
        print(f"  [Checks] Using fallback (dataflow conversion failed)")

    # Clean up temporary dataflow IR
    if os.path.exists(df_mlir):
        os.remove(df_mlir)

    # Assemble lit file: RUN lines at top, MLIR body, CHECK lines at bottom
    _assemble_lit_file(neura_mlir, dataflow_checks)

    # Stage 3: Neura → Codegen (separate verification, not in lit test)
    codegen_mlir = os.path.join(OUTPUT_DIR, f"{name}_codegen.mlir")
    print(f"  [Stage 3] Neura → Optimise + Codegen ...")
    stage3_ok = lower_neura_to_codegen(neura_mlir, codegen_mlir)
    if not stage3_ok:
        print(f"  FAIL at Stage 3 (--neura-conversion)")

    # Generate lit test copy
    if lit_dir and stage3_ok:
        lit_path = _generate_lit_test(name, neura_mlir, lit_dir)
        print(f"  [Lit]  {lit_path}")

    # Cleanup intermediate files, keep only neura.mlir
    for path in [linalg_mlir, affine_mlir, codegen_mlir]:
        if os.path.exists(path):
            os.remove(path)

    print(f"  ✓ DONE")
    return stage3_ok


def main():
    global OUTPUT_DIR
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate MLIR for all models, optionally with lit tests")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--lit", action="store_true",
                        help="Also generate lit-testable .mlir files in <output>/lit/")
    args = parser.parse_args()

    OUTPUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lit_dir = os.path.join(OUTPUT_DIR, "lit") if args.lit else None
    if lit_dir:
        os.makedirs(lit_dir, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Using neura_pipeline from: {FRONTEND_DIR}")
    if args.lit:
        print(f"Lit test directory: {lit_dir}")

    results = {}
    for name, py_file, shape in MODELS:
        results[name] = generate_model(name, py_file, shape, lit_dir=lit_dir)

    print(f"\n{'='*60}")
    print(f"  GENERATION SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {name:30s} {status}")
    print(f"\n  {passed}/{len(MODELS)} models generated")
    print(f"  Output: {OUTPUT_DIR}")
    if lit_dir:
        print(f"  Lit:    {lit_dir}")


if __name__ == "__main__":
    main()
