---
name: neura-pr-review
description: Review Neura mapping and analytical-cost-model pull requests against user annotations, PR scope, shared mapper semantics, deterministic solver behavior, and end-to-end replay correctness.
---

# Neura PR review

Use this skill for reviews or implementation follow-ups in Neura's mapping,
analytical cost model, DFG export, and CP-SAT integration.

## Review invariants

- Treat every `// USER:` annotation as a requirement: inventory it first,
  distinguish a requested explanation from a requested code change, and do not
  leave a resolved annotation in source.
- Keep the PR within its stated split. Do not pull later-PR features into it;
  shared helpers are acceptable only when they remove a genuine duplicate and
  retain the mapper's existing semantics.
- Reuse mainline mapper calculations for RecMII and ResMII. Extract a shared
  helper only when the existing mapper computation cannot otherwise be called;
  do not copy its formula into the analytical model.
- Prefer an explicit invariant failure to a guessed fallback. Unknown FU class,
  malformed shape, absent route witness, invalid coordinate, and unsupported
  payload must either have a deliberate documented mapping or fail loudly.
  A compatibility mapping for a known generic MLIR operation is not a blanket
  fallback and must state its resource semantics.
- A timeout/budget-exhausted CP-SAT result is a scheduling outcome, not proof
  of infeasibility: the agreed behavior is to try `II + 1`. Name variables for
  that meaning. Bound the solver with deterministic work, single worker, and a
  fixed seed; do not use wall-clock time as a reproducibility constraint.
- Exact mapping means a complete witness. CP-SAT must jointly constrain
  placement, schedule, and routing; import must require and replay all routes,
  never silently hand a placement-only witness to the heuristic router.
- Preserve the mapper's producer fanout semantics in a joint route model: one
  time-expanded graph and one set of link/register/port reservations per
  producer value, with a path emitted for every consumer.  Per-edge route
  graphs double-count broadcasts and create avoidable scalability failures.
- Verify pipeline wiring end to end: analytical mode must serialize the same
  placed-op order and DFG/architecture contract that the importer consumes.
  DFG edge construction, architecture-shape parsing, and operation/FU
  classification must have one source of truth.
- For changed e2e behavior, refresh tests only from a real temporary artifact.
  Keep a separate prefix for analytical output and check the complete generated
  artifact, not selected metadata. Record environmental/toolchain and solver
  discoveries in the relevant skill.

## Review output

Report findings first, ordered by severity, with file/line evidence and a
concrete consequence. Then list verified requirements and residual validation
limits. Do not commit or mutate code during a review unless separately asked.
