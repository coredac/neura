#!/usr/bin/env python3
## @file generate_mlir.py
## @brief Generate lit-testable MLIR files for all PyTorch models in models/.
##
## @details
## This script iterates over the model list defined in #MODELS and runs the
## full neura_pipeline compilation flow for each model:
##   - Stage 0: PyTorch -> Linalg MLIR
##   - Stage 1: Linalg -> Affine MLIR
##   - Stage 2: Affine -> Taskflow -> Neura MLIR
##   - Strip: Remove taskflow.task wrappers to obtain clean Neura IR
##   - Dataflow: Neura IR -> Dataflow IR (for structural checking)
##   - Interpreter: Execute dataflow IR and capture output (for value checking)
##   - Stage 3: Neura -> Optimise + Codegen (standalone verification, not in lit)
##
## Each model produces a self-contained <name>_neura.mlir lit test with:
##   1. RUN directives at the top to drive the pipeline + FileCheck
##   2. CHECK-DAG / CHECK-NOT patterns at the bottom for structural & value checks
##
## When --lit is passed, a copy is also placed under <output-dir>/lit/.
##
## @usage
##   python generate_mlir.py [--output-dir generated] [--lit]
##
## @note Every model .py file under models/ must expose an nn.Module instance
##       named `model`.

import os
import re
import subprocess
import sys
import shutil

## @var SCRIPT_DIR
## @brief Absolute path to the directory containing this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

## @var FRONTEND_DIR
## @brief Absolute path to the neura-py-frontend tools directory.
## @details Added to sys.path so that neura_pipeline can be imported.
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

## @var MODELS_DIR
## @brief Directory containing per-model .py definitions.
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

## @var OUTPUT_DIR
## @brief Default output directory for generated .mlir artifacts.
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated")

## @var LIT_DIR
## @brief Sub-directory under OUTPUT_DIR where lit-only .mlir copies are placed.
LIT_DIR = os.path.join(OUTPUT_DIR, "lit")

## @var MODELS
## @brief List of models to generate.
## @details Each tuple contains (output_name, python_module_name, example_shape).
##          The corresponding .py file under MODELS_DIR must expose an `model`
##          attribute of type torch.nn.Module.
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

## @var DATAFLOW_PASSES
## @brief Ordered list of mlir-neura-opt passes for the dataflow conversion pipeline.
## @details Equivalent to --neura-conversion but stops before --map-to-accelerator
##          and --generate-code, producing pure dataflow IR for structural checks.
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

## @var _ALL_COMPUTE_OPS
## @brief Complete list of neura compute ops that may appear in dataflow IR.
## @details Used by _generate_dataflow_checks() to auto-detect which ops
##          are present and emit the corresponding CHECK-DAG lines.
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


## @brief Locate the mlir-neura-opt binary.
## @details Searches the build directory first, then falls back to PATH.
## @return Absolute path to mlir-neura-opt, or the bare name if not found locally.
def _find_opt() -> str:
    build_dir = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "build")
    candidate = os.path.join(build_dir, "tools", "mlir-neura-opt", "mlir-neura-opt")
    if os.path.exists(candidate):
        return candidate
    return "mlir-neura-opt"


## @brief Locate the neura-interpreter binary.
## @details Searches the build directory first, then falls back to PATH.
## @return Absolute path to neura-interpreter, or the bare name if not found locally.
def _find_interpreter() -> str:
    build_dir = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "build")
    candidate = os.path.join(
        build_dir, "tools", "neura-interpreter", "neura-interpreter"
    )
    if os.path.exists(candidate):
        return candidate
    return "neura-interpreter"


## @brief Run the dataflow conversion pass pipeline on Neura IR.
## @param neura_mlir Path to the clean Neura IR .mlir file (no lit header).
## @param output_mlir Path to write the resulting dataflow IR .mlir file.
## @return True on success, False on failure (stderr is printed).
def _run_dataflow_passes(neura_mlir: str, output_mlir: str) -> bool:
    opt = _find_opt()
    result = subprocess.run(
        [opt] + DATAFLOW_PASSES + [neura_mlir, "-o", output_mlir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [dataflow] FAILED:\n{result.stderr}", file=sys.stderr)
        return False
    return True


## @brief Execute neura-interpreter on a dataflow MLIR file.
## @param dataflow_mlir Path to the dataflow IR .mlir file.
## @return stdout string on success, or None on failure (stderr is printed).
def _run_dataflow_interpreter(dataflow_mlir: str):
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


## @brief Generate FileCheck patterns from dataflow IR and interpreter output.
## @param dataflow_mlir Path to the dataflow IR .mlir file.
## @param interpreter_output stdout captured from neura-interpreter.
## @return String containing DATAFLOW_IR-DAG and INTERPRETER_OUTPUT-DAG/NOT lines.
## @details
## Structural checks (DATAFLOW_IR-DAG):
##   - module with torch.debug_module_name
##   - func.func @forward with dataflow_mode="predicate"
##   - neura.kernel, neura.counter, neura.load_indexed, neura.store_indexed
##   - Compute ops auto-detected from #_ALL_COMPUTE_OPS
##   - neura.yield
##
## Value checks (INTERPRETER_OUTPUT-DAG):
##   - "Store to m" for store existence
##   - "Output: <value>" for the deterministic final output
##
## Error checks (INTERPRETER_OUTPUT-NOT):
##   - Error, Failed, Unhandled
def _generate_dataflow_checks(dataflow_mlir: str, interpreter_output: str) -> str:
    checks = []

    # --- Read the dataflow IR content ---
    with open(dataflow_mlir) as f:
        df_content = f.read()

    # --- DATAFLOW_IR: structural checks using CHECK-DAG ---
    # DAG ordering avoids sensitivity to operation order in the generated IR.
    checks.append('// DATAFLOW_IR-DAG: module attributes {torch.debug_module_name')
    checks.append('// DATAFLOW_IR-DAG: func.func @forward')
    checks.append('// DATAFLOW_IR-DAG: dataflow_mode = "predicate"')
    checks.append("// DATAFLOW_IR-DAG: neura.kernel")
    checks.append("// DATAFLOW_IR-DAG: neura.counter")
    checks.append("// DATAFLOW_IR-DAG: neura.load_indexed")
    checks.append("// DATAFLOW_IR-DAG: neura.store_indexed")

    # Auto-detect present compute ops, dropping sub-string matches
    present_ops = [op for op in _ALL_COMPUTE_OPS if op in df_content]
    selected = []
    for op in present_ops:
        if any(longer for longer in present_ops if longer != op and op in longer):
            continue
        selected.append(op)
    for op_name in selected:
        checks.append(f"// DATAFLOW_IR-DAG: {op_name}")

    checks.append("// DATAFLOW_IR-DAG: neura.yield")

    # --- INTERPRETER_OUTPUT: value checks + no-error checks ---
    if interpreter_output:
        if re.search(r'Store to m\d+/', interpreter_output):
            checks.append("// INTERPRETER_OUTPUT-DAG: Store to m")

        for line in interpreter_output.split("\n"):
            m = re.search(r'Output:\s*([-\d.e+]+)', line)
            if m:
                checks.append(f"// INTERPRETER_OUTPUT-DAG: Output: {m.group(1)}")
                break

        checks.append("// INTERPRETER_OUTPUT-NOT: Error")
        checks.append("// INTERPRETER_OUTPUT-NOT: Failed")
        checks.append("// INTERPRETER_OUTPUT-NOT: Unhandled")

    return "\n".join(checks) + "\n"


## @brief Build the lit RUN directives for the dataflow + interpreter test.
## @return String containing RUN lines to place at the top of the .mlir file.
## @details
## The RUN pipeline consists of four steps:
##   1. mlir-neura-opt: convert Neura IR -> dataflow IR
##   2. neura-interpreter: execute dataflow IR, capture output
##   3. FileCheck --check-prefix=DATAFLOW_IR: verify dataflow IR structure
##   4. FileCheck --check-prefix=INTERPRETER_OUTPUT: verify interpreter output
def _lit_header() -> str:
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


## @brief Wrap CHECK lines in a footer block to append at the end of the file.
## @param dataflow_checks Pre-generated CHECK lines from _generate_dataflow_checks().
## @return String with a leading newline followed by the CHECK lines.
def _lit_footer(dataflow_checks: str) -> str:
    return "\n" + dataflow_checks


## @brief Produce minimal fallback CHECK lines when dataflow conversion fails.
## @return String containing a basic DATAFLOW_IR check and INTERPRETER_OUTPUT-NOT.
def _dataflow_fallback_checks() -> str:
    return (
        "// DATAFLOW_IR: neura.kernel\n"
        "// INTERPRETER_OUTPUT-NOT: Error\n"
    )


## @brief Rewrite a .mlir file in-place with lit RUN header and CHECK footer.
## @param neura_mlir Path to the Neura IR .mlir file to modify.
## @param dataflow_checks CHECK lines to append at the bottom.
def _assemble_lit_file(neura_mlir: str, dataflow_checks: str) -> None:
    with open(neura_mlir) as f:
        content = f.read()
    with open(neura_mlir, "w") as f:
        f.write(_lit_header())
        f.write(content)
        f.write(_lit_footer(dataflow_checks))


## @brief Copy a headered Neura .mlir file into the lit test directory.
## @param name Base name for the lit test file (without extension).
## @param neura_mlir Path to the source .mlir file (already with lit header/footer).
## @param lit_dir Target directory for the lit copy.
## @return Path to the copied lit .mlir file.
def _generate_lit_test(name: str, neura_mlir: str, lit_dir: str) -> str:
    lit_mlir = os.path.join(lit_dir, f"{name}.mlir")
    shutil.copy(neura_mlir, lit_mlir)
    return lit_mlir


## @brief Strip taskflow.task wrappers from a Neura .mlir file in-place.
## @param neura_mlir Path to the Neura IR .mlir file to modify.
## @return True on success, False on failure.
def _strip_taskflow(neura_mlir: str) -> bool:
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


## @brief Run the full compilation pipeline for a single model.
## @param name Output name used for generated .mlir files.
## @param py_file Basename (without .py) of the model definition under models/.
## @param example_shape Input tensor shape as a list of ints.
## @param lit_dir Optional lit output directory; if set, a lit copy is produced.
## @return True if Stage 3 (codegen) succeeded, False otherwise.
## @details Pipeline stages:
##   - Stage 0: export_pytorch_to_linalg()
##   - Stage 1: lower_linalg_to_affine()
##   - Stage 2: lower_affine_to_neura()
##   - Strip:   _strip_taskflow()
##   - Dataflow + Interpreter checks for lit file assembly
##   - Stage 3: lower_neura_to_codegen()
##   - Cleanup: remove intermediate .mlir files, keep only <name>_neura.mlir
def generate_model(name: str, py_file: str, example_shape: list,
                   lit_dir: str = None) -> bool:
    print(f"\n{'='*60}")
    print(f"  {name}  (shape={example_shape})")
    print(f"{'='*60}")

    model_path = os.path.join(MODELS_DIR, f"{py_file}.py")
    if not os.path.exists(model_path):
        print(f"  SKIP: model file not found: {model_path}")
        return False

    # Stage 0: PyTorch -> Linalg
    linalg_mlir = os.path.join(OUTPUT_DIR, f"{name}_linalg.mlir")
    print(f"  [Stage 0] PyTorch -> Linalg ...")
    if not export_pytorch_to_linalg(model_path, linalg_mlir, example_shape=example_shape):
        print(f"  FAIL at Stage 0")
        return False

    # Stage 1: Linalg -> Affine
    affine_mlir = os.path.join(OUTPUT_DIR, f"{name}_affine.mlir")
    print(f"  [Stage 1] Linalg -> Affine ...")
    if not lower_linalg_to_affine(linalg_mlir, affine_mlir):
        print(f"  FAIL at Stage 1")
        return False

    # Stage 2: Affine -> Taskflow -> Neura
    neura_mlir = os.path.join(OUTPUT_DIR, f"{name}_neura.mlir")
    print(f"  [Stage 2] Affine -> Taskflow -> Neura ...")
    if not lower_affine_to_neura(affine_mlir, neura_mlir):
        print(f"  FAIL at Stage 2")
        return False

    # Strip taskflow.task wrappers
    print(f"  [Strip] Removing taskflow.task wrappers ...")
    if not _strip_taskflow(neura_mlir):
        print(f"  FAIL at strip-taskflow-task")
        return False

    # Generate dataflow IR + interpreter output for CHECK lines
    df_mlir = os.path.join(OUTPUT_DIR, f"{name}_dataflow.mlir")
    print(f"  [Dataflow] Converting Neura -> Dataflow ...")
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

    # Assemble lit file: RUN at top, MLIR body, CHECK at bottom
    _assemble_lit_file(neura_mlir, dataflow_checks)

    # Stage 3: Neura -> Optimise + Codegen (standalone verification)
    codegen_mlir = os.path.join(OUTPUT_DIR, f"{name}_codegen.mlir")
    print(f"  [Stage 3] Neura -> Optimise + Codegen ...")
    stage3_ok = lower_neura_to_codegen(neura_mlir, codegen_mlir)
    if not stage3_ok:
        print(f"  FAIL at Stage 3 (--neura-conversion)")

    # Generate lit test copy if requested
    if lit_dir and stage3_ok:
        lit_path = _generate_lit_test(name, neura_mlir, lit_dir)
        print(f"  [Lit]  {lit_path}")

    # Cleanup intermediate files, keep only neura.mlir
    for path in [linalg_mlir, affine_mlir, codegen_mlir]:
        if os.path.exists(path):
            os.remove(path)

    print(f"  DONE")
    return stage3_ok


## @brief Entry point: parse arguments and generate MLIR for all models.
##
## @details
## CLI arguments:
##   --output-dir <path>   Output directory (default: generated/)
##   --lit                 Also produce a copy in <output-dir>/lit/
##
## After generation, a summary table is printed showing pass/fail per model.
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
        status = "OK" if ok else "FAIL"
        print(f"  {name:30s} {status}")
    print(f"\n  {passed}/{len(MODELS)} models generated")
    print(f"  Output: {OUTPUT_DIR}")
    if lit_dir:
        print(f"  Lit:    {lit_dir}")


if __name__ == "__main__":
    main()
