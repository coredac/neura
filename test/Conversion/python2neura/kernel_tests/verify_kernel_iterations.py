#!/usr/bin/env python3
## @file verify_kernel_iterations.py
## @brief Verify neura.kernel counter iteration counts.
##
## Validates counter behavior by inspecting interpreter verbose output:
##   - leaf-only:   upper=4 → 4 iterations
##   - nested:      root=4, relay=3, leaf=2 → 4*3*2 = 24 iterations
##   - single_iter: upper=1 → 1 iteration
##   - step=2:      upper=6, step=2 → ceil(6/2) = 3 iterations
##
## Usage:
## @code{.sh}
## python3 verify_kernel_iterations.py
## @endcode

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
INTERPRETER = os.path.join(WORKSPACE, "build/tools/neura-interpreter/neura-interpreter")
TEST_FILE = os.path.join(SCRIPT_DIR, "kernel_unit_tests.mlir")


def run_verbose(mode_args=None):
    ## @brief Run interpreter in verbose mode and return combined output.
    ## @param mode_args Optional list of extra CLI arguments (e.g. ["--dataflow"]).
    ## @return stdout + stderr as a single string.
    cmd = [INTERPRETER, TEST_FILE, "--verbose"]
    if mode_args:
        cmd.extend(mode_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr


def count_iterations(output, counter_id):
    ## @brief Count how many times a counter printed its index.
    ## @param output The raw verbose output from the interpreter.
    ## @param counter_id The counter identifier (currently unused; left for future filtering).
    ## @return Total number of "Counter index:" lines found.
    pattern = re.compile(r"Counter index: (\d+)")
    matches = pattern.findall(output)
    if not matches:
        return 0
    return len(matches)


def test_cf_mode():
    ## @brief Run control-flow mode iteration tests.
    ## @return True if all checks pass, False otherwise.
    print("=== Control-flow Mode Iteration Tests ===")
    output = run_verbose()

    # Count total Counter index lines (all counters across all functions)
    total_counter_lines = len(re.findall(r"Counter index:", output))
    print(f"  Total Counter index lines: {total_counter_lines}")

    # Per-function counts:
    # test2: leaf counter (upper=4) → 4 iterations → 4 "Counter index:" lines
    # test3: 3 counters nested (root=4,relay=3,leaf=2) → 24 iterations, 3 × 24 = 72 lines
    # test4: leaf (upper=1) → 1 iteration → 1 line
    # test5: leaf (upper=6,step=2) → 3 iterations → 3 lines
    # test6: 2 kernels, leaf=3 + leaf=5 → 3+5 = 8 lines
    # test7: kernel A (2×2×2=8) + kernel B (3×2×2=12) → 3×8 + 3×12 = 24+36=60 lines
    # test8: 4 kernels, each leaf(upper=2) → 4×2 = 8 lines
    #
    # Expected total: 4 + 72 + 1 + 3 + 8 + 60 + 8 = 156
    expected_total = 4 + 72 + 1 + 3 + 8 + 60 + 8

    print(f"  Expected Counter index lines: {expected_total}")
    if total_counter_lines == expected_total:
        print("  ✓ PASS: Iteration count matches expected")
    else:
        print(f"  ✗ FAIL: Expected {expected_total}, got {total_counter_lines}")
        return False

    # Verify no errors
    if "Error" in output or "Failed" in output:
        print("  ✗ FAIL: Found error in output")
        return False
    print("  ✓ No errors in output")
    return True


def test_df_mode():
    ## @brief Run dataflow mode iteration tests.
    ## @return True if all checks pass, False otherwise.
    print("\n=== Dataflow Mode Iteration Tests ===")
    output = run_verbose(["--dataflow"])

    # In dataflow mode, each DFG iteration prints "DFG Iteration N"
    dfg_iterations = re.findall(r"DFG Iteration \d+ - Beginning", output)
    print(f"  Total DFG Iteration lines: {len(dfg_iterations)}")

    # Count counters initialized
    counter_inits = re.findall(r"Counter initialized:", output)
    print(f"  Total counters initialized: {len(counter_inits)}")

    # Verify no errors
    if "Error" in output or "Failed" in output:
        print("  ✗ FAIL: Found error in output")
        return False
    print("  ✓ No errors in output")

    # Verify all 8 outputs are present
    outputs = re.findall(r"Output: (\d+\.\d+)", output)
    expected_outputs = ["1.000000", "2.000000", "3.000000", "4.000000",
                        "5.000000", "6.000000", "7.000000", "8.000000"]
    for eo in expected_outputs:
        if eo in outputs:
            print(f"  ✓ Found expected output: {eo}")
        else:
            print(f"  ✗ Missing expected output: {eo}")
            return False
    return True


def main():
    ## @brief Entry point: run CF and DF tests, print summary, exit with status.
    ## @return None (exits via sys.exit).
    results = []
    results.append(("Control-flow mode", test_cf_mode()))
    results.append(("Dataflow mode", test_df_mode()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:<30} {status}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
