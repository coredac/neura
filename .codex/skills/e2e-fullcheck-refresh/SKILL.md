---
name: e2e-fullcheck-refresh
description: Update Neura e2e FileCheck expectations from generated temporary outputs, preserving bounded generated-output coverage and recording failures discovered during the refresh.
---

# E2E full-check refresh

Use this skill when a Neura e2e pipeline changes and its checked output must be
refreshed from an actual generated temporary artifact.

## Required workflow

1. Run the exact `RUN:` pipeline from the test, redirecting its full stdout
   and stderr to a uniquely named file under `/tmp`. Do not synthesize expected
   output from source code or hand-edit individual values.
2. Copy the generated artifact from that file into the test's matching check
   block, using the test's existing FileCheck style. Keep every retained line
   literal and give it the appropriate check prefix (`PREFIX:`,
   `PREFIX-NEXT:`, or `PREFIX-LABEL:`); do not reduce the retained block to key
   fields such as `compiled_ii` or mask differences with broad wildcards. Apply
   these output-size limits:
   - For mapping MLIR, retain and check at most the first 200 emitted lines.
   - For YAML or ASM, retain and check all emitted lines when there are at most
     50; when there are more than 50, retain and check only the first 50.
   These limits intentionally leave any later emitted lines unchecked; do not
   claim that the omitted suffix was checked or invent a trailer for it.
3. If the test needs a new pipeline mode, add a dedicated check prefix and a
   `RUN:` line for that mode. Keep its expected output separate from a
   heuristic baseline so the comparison remains readable.
4. Run the individual test after each refresh. For every failure, inspect the
   saved `/tmp` artifact and either fix the implementation/test expectation or
   add a concise, reusable discovery to this skill under **Discoveries**.
5. Before handoff, rerun every changed e2e test and verify each retained check
   segment is generated from its corresponding `/tmp` artifact. Confirm that
   the segment ends at the applicable limit when the artifact exceeds it.

Do not commit, delete unrelated artifacts, or weaken the retained-output check
to make a test pass. The limits above are the only permitted omission; do not
substitute selected metadata or broad wildcards for retained artifact lines.

## Discoveries

- The global Codex skill directory can be read-only in this environment; this
  repository-local copy is the usable skill source for the current worktree.
- The e2e `RUN:` lines invoke `llvm-extract` through `PATH`, while the generated
  lit configuration only substitutes several other LLVM tools.  Prepend
  `/home/x/shiran/llvm-project/build/bin` when refreshing locally; otherwise
  the system LLVM 16 `llvm-extract` rejects IR emitted by LLVM 20.
- The system `python3` has no `ortools`, but the existing isolated verifier
  environment `/tmp/neura-pr1-verify-venv/bin/python` provides OR-Tools
  9.11.4210.  Pass it explicitly as
  `python-executable=/tmp/neura-pr1-verify-venv/bin/python` to exercise a
  dynamic analytical e2e run; do not silently fall back to static imports.
- If an existing heuristic e2e aborts at
  `mapping_util.cpp:getOperationKindFromMlirOp` with
  `unknown MLIR operation has no Neura OperationKind`, stop expectation
  refresh: this is an implementation regression before FileCheck runs, not an
  output change.  Capture the full lit failure in a unique `/tmp` log and fix
  the mapper before generating analytical check blocks.
- Do not reintroduce a blanket `IAdd` fallback to repair that regression.
  Some e2e textual Neura grant operations arrive as generic operations, so
  their names must be recognized explicitly; independently, every operation
  accepted by `occupiesFU` needs a deliberate `OperationKind` (for example,
  `PhiStartOp` should have the same kind as `PhiOp`) or be excluded from
  physical placement.  The original AXPY reproducer is
  `/tmp/neura-e2e-axpy-heuristic-regression.log`.
- The joint CP-SAT model's compactness objective can take more wall-clock time
  than its per-solve deterministic budget suggests because it runs multiple
  solver phases.  AXPY succeeded at II=5 after roughly 74 seconds (two
  approximately-30-second phases plus replay), with a compact t=0..9 witness.
  Use a PTY session and poll it rather than a one-shot 30-second command; the
  latter may terminate before the mapping artifact is written.  The generated
  compact AXPY artifact is `/tmp/neura-e2e-axpy-analytical-compact3.mlir`.
- A reduced deterministic budget can return an early feasible joint witness
  before the compactness objective completes.  With the final 12-unit budget,
  AXPY and FIR still reached II=5, but their generated MLIR became 11KB and
  15KB respectively because routes were no longer compact.  Before copying a
  dynamic output verbatim into FileCheck, rerun it with the exact test budget
  and compare the artifacts; if it differs, fix/canonicalize the solver output
  or use a budget that completes the objective rather than committing a flaky
  full-output check.
- The all-e2e baseline currently aborts before FileCheck on GEMM in
  `calculateRouteMii`: a `DataMovOp` can carry a non-scalar payload, so a
  scalar-only `valueBitWidth` assertion is an implementation bug, not an
  expectation drift.  Preserve `/tmp/neura-e2e-baseline-all.log`, fix/rebuild
  the mapper, then regenerate every affected artifact.
- A concurrently interrupted relink can leave
  `build/tools/mlir-neura-opt/mlir-neura-opt` as a zero-byte, non-executable
  file.  In that state every lit e2e test fails with exit 126 before mapping;
  do not refresh checks.  Rebuild the target and verify the executable is
  non-empty before rerunning the suite.
- Generated mapping MLIR has one significant final blank line after the module
  closing brace.  Preserve it with `PREFIX-EMPTY:` after the final
  `PREFIX-NEXT: }` when it falls within the first 200 emitted lines.  If the
  artifact exceeds that mapping limit, stop at line 200 and do not claim that
  the suffix (including any later blank line) was checked.
- Do not add a dynamic analytical lit test for a workload until its exact
  pipeline has completed with the test's final deterministic budget.  In the
  current joint model, BICG-int produced no mapping after more than seven
  minutes at the default 12-unit budget; the interrupted full log is
  `/tmp/neura-e2e-bicg-int-analytical-final.log`.  This is a solver
  scalability/option-design issue, not an excuse to create a partial check or
  substitute a static imported witness.
- `max_deterministic_time` bounds CP-SAT search work only; it does not bound
  the host work required to construct the time-expanded joint-routing model.
  Before launching CP-SAT, use the input-derived route-node limit and report a
  `model limit` result when it is exceeded. This is deterministic across
  servers and is not a heuristic fallback. Do not create a successful
  analytical check from a workload that hits this limit; add an explicit
  expected-failure test only when that behavior is the intended contract.
- `neura.vector.reduce.add` must have an explicit OperationKind (currently
  IAdd) for FIR-vector heuristic mapping.  Until the mapper is rebuilt after
  this addition, e2e fails before producing the mapping artifact; regenerate
  both baseline and analytical outputs only after that rebuild.
- A second AXPY analytical lit run with the same toolchain and fixed CP-SAT
  seed produced the same mapping body but a different ordering of entries in
  the printed `dlti.dl_spec` module attribute.  A literal whole-line check of
  that attribute is therefore not reproducible.  Treat this as a printer/IR
  canonicalization issue: preserve literal checks for the retained mapping
  body, but do not claim a test is stable until the module attribute is
  canonicalized or the check format can represent all retained entries
  independent of their order.
- The mapper now canonicalizes `dlti.dl_spec` at the end of mapping by sorting
  its entries before printing.  Refresh mapping checks only after this pass is
  present; two AXPY automatic analytical runs then produced bytewise identical
  MLIR.
- Do not depend on the current working directory for analytical e2e tests.
  The CMake build packages `exact_mapper_cpsat.py` at
  `build/share/neura/exact_mapper_cpsat.py` and the pass selects it by default;
  tests should override only the Python interpreter needed to provide the
  pinned OR-Tools wheel.
- Do not hard-code the verifier venv path in a committed `RUN:` line.
  `test/lit.cfg.in` exposes `%neura_python` from `NEURA_PYTHON_EXECUTABLE` and
  adds the `neura-ortools` feature only when that interpreter imports OR-Tools.
  Configure the build with a pinned interpreter that has OR-Tools; tests then
  use `REQUIRES: neura-ortools` and `python-executable=%neura_python`.
