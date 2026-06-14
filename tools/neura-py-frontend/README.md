# Neura Python Frontend

Compile PyTorch `nn.Module` to Neura CGRA dataflow IR, with full lowering pipeline and end-to-end numerical validation.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Frontend Pipeline](#frontend-pipeline)
- [New Pass Design](#new-pass-design)
- [Interpreter Implementation](#interpreter-implementation)
- [Arithmetic Lowering](#arithmetic-lowering)
- [Testing](#testing)
- [Dependencies](#dependencies)

---

## Quick Start

### Environment Setup

```bash
# 1. Create conda environment
conda env create -f tools/neura-py-frontend/environment.yml
conda activate neura-torch

# 2. Build the project
mkdir -p build && cd build
cmake .. -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
         -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm
make -j$(nproc) mlir-neura-opt neura-interpreter neura-compiler neura-py-frontend
```

Key binaries after build:

| Binary | Purpose |
|--------|---------|
| `build/tools/mlir-neura-opt/mlir-neura-opt` | MLIR pass driver |
| `build/tools/neura-interpreter/neura-interpreter` | Neura IR interpreter |
| `build/tools/neura-compiler/neura-compiler` | Compiler frontend |
| `build/tools/neura-py-frontend/neura-py-frontend` | Convenience shell wrapper |

### CLI Usage

```bash
# Full pipeline from a PyTorch model
python tools/neura-py-frontend/neura_pipeline.py \
    --model path/to/model.py \
    --example-shape 4 8 \
    --output output.json

# From existing Linalg-on-Tensors .mlir
python tools/neura-py-frontend/neura_pipeline.py \
    --input linalg_ir.mlir \
    --output output.json

# Stop after Stage 2 (Neura dialect, before codegen)
python tools/neura-py-frontend/neura_pipeline.py \
    --input linalg_ir.mlir \
    --output output.mlir \
    --stop-after taskflow

# With CGRA architecture spec
python tools/neura-py-frontend/neura_pipeline.py \
    --model model.py \
    --example-shape 4 8 \
    --arch-spec test/arch_spec/architecture.yaml \
    --output output.json
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Path to `.py` file exposing a `model` attribute |
| `--input`, `-i` | Path to existing Linalg-on-Tensors `.mlir` |
| `--output`, `-o` | Output path (default: `output.json`) |
| `--stop-after` | Stop after: `affine`, `taskflow`, or `neura` (default) |
| `--keep-intermediate` | Keep intermediate `.mlir` files on disk |
| `--func-name` | Function name for torch-mlir export (default: `forward`) |
| `--example-shape` | Example input shape, e.g. `1 3 224 224` |
| `--arch-spec` | Path to CGRA architecture spec YAML |

`--model` and `--input` are mutually exclusive.

### Programmatic API

```python
from neura_pipeline import (
    export_pytorch_to_linalg,   # PyTorch .py → Linalg-on-Tensors .mlir
    lower_linalg_to_affine,     # Linalg → Affine
    lower_affine_to_neura,      # Affine → Neura
    lower_neura_to_codegen,     # Neura → CGRA Codegen
    full_pipeline,              # Linalg → Codegen in one call
)

# Step by step
export_pytorch_to_linalg("model.py", "linalg.mlir", example_shape=[1, 3, 224, 224])
lower_linalg_to_affine("linalg.mlir", "affine.mlir")
lower_affine_to_neura("affine.mlir", "neura.mlir")
lower_neura_to_codegen("neura.mlir", "output.json")

# Or all at once
full_pipeline(
    input_mlir="linalg.mlir",
    output_file="output.json",
    arch_spec=None,
    stop_after=None,
    keep_intermediate=False,
)
```

### Model File Convention

Each model `.py` must expose a `torch.nn.Module` instance named `model`:

```python
import torch
import torch.nn as nn

class SimpleMatMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        return torch.matmul(x, self.weight)

model = SimpleMatMul()
```

8 built-in test models under `test/Conversion/python2neura/model_tests/models/`:
`simple_matmul`, `residual_block`, `residual_block_norelu`, `two_layer_mlp`,
`two_layer_mlp_norelu`, `conv2d_relu_pool`, `transformer_attention`, `gelu_layernorm`.

---

## Architecture

### Overall Flow

```mermaid
flowchart TD
    A["PyTorch nn.Module (.py)"] -->|"torch-mlir: export_and_import"| B["Linalg-on-Tensors (.mlir)"]
    B -->|"Stage 1: linalg-to-affine-conversion"| C["Affine (.mlir)"]
    C -->|"Stage 2: taskflow-conversion"| D["Neura Dialect (.mlir)"]
    D -->|"Stage 3: neura-conversion"| E["Optimised Neura + CGRA JSON"]
```

### File Layout

| File | Purpose |
|------|---------|
| `neura_pipeline.py` | Python API + CLI for the full lowering pipeline |
| `neura-py-frontend.sh` | Shell convenience wrapper |
| `CMakeLists.txt` | Installs scripts into build output directory |
| `environment.yml` | Conda environment (torch + torch-mlir) |

Tool auto-discovery: `neura_pipeline.py` resolves `mlir-neura-opt` by checking the build directory relative to the script location first, then falls back to system `PATH`.

---

## Frontend Pipeline

The pipeline consists of 4 stages:

```mermaid
flowchart LR
    subgraph Stage0["Stage 0: Export"]
        direction LR
        S0A["model.py<br/>nn.Module"] --> S0B["torch-mlir<br/>export_and_import"]
        S0B --> S0C["Linalg IR"]
    end

    subgraph Stage1["Stage 1: Lowering"]
        direction LR
        S1A["Linalg IR"] --> S1B["Bufferize<br/>Linalg→Affine"]
        S1B --> S1C["Expand<br/>Math→Arith"]
        S1C --> S1D["Affine IR"]
    end

    subgraph Stage2["Stage 2: Taskflow"]
        direction LR
        S2A["Affine IR"] --> S2B["Affine→Taskflow"]
        S2B --> S2C["Taskflow→Neura"]
        S2C --> S2D["Neura IR"]
    end

    subgraph Stage3["Stage 3: Codegen"]
        direction LR
        S3A["Neura IR"] --> S3B["Strip Task"]
        S3B --> S3C["Pattern Fuse<br/>+ Data Mov"]
        S3C --> S3D["CGRA JSON"]
    end

    Stage0 --> Stage1 --> Stage2 --> Stage3
```

### Stage 0: PyTorch → Linalg-on-Tensors

Implemented in `export_pytorch_to_linalg()`:

```python
from torch_mlir.fx import export_and_import, OutputType

mlir_module = export_and_import(
    model, example_input,
    output_type=OutputType.LINALG_ON_TENSORS,
    func_name="forward",
)
```

**Key adaptation for torch-mlir 20260531:** The old `torch_mlir.compile()` was deprecated. The new API uses:
- `torch_mlir.fx.export_and_import()` instead of `compile()`
- `OutputType.LINALG_ON_TENSORS` enum instead of the `"linalg-on-tensors"` string

Models are loaded dynamically via `importlib`, requiring a `model` attribute of type `nn.Module`.

### Stage 1: Linalg → Affine

Registered in `lib/TaskflowDialect/TaskflowPasses.cpp` as `--linalg-to-affine-conversion`:

```mermaid
flowchart TD
    P1["linalg-generalize-named-ops"] --> P2["eliminate-empty-tensors"]
    P2 --> P3["canonicalize"]
    P3 --> P4["one-shot-bufferize<br/>(tensor→memref)"]
    P4 --> P5["convert-linalg-to-affine-loops"]
    P5 --> P6["fold-memref-alias-ops"]
    P6 --> P7["fold-affine-subview"]
    P7 --> P8["convert-copy-to-affine-loops"]
    P8 --> P9["affine-loop-normalize"]
    P9 --> P10["simplify-affine-structures"]
    P10 --> P11["canonicalize"]
    P11 --> P12["expand-math-to-arith<br/>(★ new pass)"]
```

The final `expand-math-to-arith` pass is critical — it expands `math.fpowi` and `math.tanh` into `arith` ops before they get promoted into kernel block arguments, where constant exponents would be hidden.

### Stage 2: Affine → Taskflow → Neura

Registered in `lib/TaskflowDialect/TaskflowPasses.cpp` as `--taskflow-conversion`:

```
convert-affine-to-taskflow → construct-hyperblock-from-task
→ classify-task-and-counter → convert-taskflow-to-neura
```

Output is Neura dialect IR. Use `--stop-after taskflow` to intercept here.

### Stage 3: Neura → Optimised + CGRA Codegen

Registered in `lib/NeuraDialect/NeuraPasses.cpp` as `--neura-conversion`:

```mermaid
flowchart TD
    subgraph S3_Clean["Cleanup"]
        A["strip-taskflow-task ★"] --> B["assign-accelerator"]
    end

    subgraph S3_Lower["Dialect Lowering"]
        C1["lower-affine-to-neura"]
        C2["lower-arith-to-neura"]
        C3["lower-memref-to-neura"]
        C4["lower-builtin-to-neura"]
        C5["lower-llvm-to-neura"]
    end

    subgraph S3_Opt["Optimisations"]
        D1["canonicalize-return"] --> D2["canonicalize-cast"]
        D2 --> D3["promote-input-arg-to-const"]
        D3 --> D4["fold-constant"]
        D4 --> D5["canonicalize-live-in"]
    end

    subgraph S3_Dataflow["Dataflow Transformation"]
        E1["leverage-predicated-value"] --> E2["transform-ctrl-to-data-flow"]
        E2 --> E3["fold-constant"]
    end

    subgraph S3_Codegen["Codegen"]
        F1["fuse-pattern"] --> F2["insert-data-mov"]
        F2 --> F3["map-to-accelerator"]
        F3 --> F4["generate-code"]
    end

    A --> B
    B --> C1 & C2 & C3 & C4 & C5
    C1 & C2 & C3 & C4 & C5 --> D1
    D5 --> E1
    E3 --> F1
```

All three stages can also be folded into a single pipeline via `--python-to-neura`.

---

## New Pass Design

### `expand-math-to-arith`: Pre-expand Math Ops

**File:** `lib/Conversion/ExpandMathToArith/ExpandMathToArithPass.cpp`

**Motivation:** Stage 2's taskflow conversion promotes constants to kernel block arguments. Once promoted, expressions like `x^3` (used in GELU's tanh approximation) no longer see a visible constant exponent, making expansion into `arith.mulf` chains impossible. This pass runs before affine-loop lowering to guarantee all exponents are constant-foldable.

```mermaid
flowchart LR
    subgraph Problem["Without expand-math-to-arith"]
        P_A["math.fpowi %x, 3"] --> P_B["Taskflow promotes\nconstant to arg"]
        P_B -.->|"✗ exponent lost"| P_C["arith.mulf\n%x, %x, %x"]
    end

    subgraph Solution["With expand-math-to-arith"]
        S_A["math.fpowi %x, 3"] --> S_B["arith.mulf\n%x, %x, %x"]
        S_B --> S_C["lowered to\nneura.mulf"]
    end
```

**Patterns:**

| Pattern | Input | Output |
|---------|-------|--------|
| `ExpandFPowI` | `math.fpowi(x, const_n)` n≥1 | `x * x * ... * x` (n times) |
| `ExpandTanh` | `math.tanh(x)` | `(exp(2x) - 1) / (exp(2x) + 1)` |

These decomposed ops are later lowered to `neura.mulf`, `neura.exp`, `neura.subf`, `neura.divf` by `ArithToNeura`.

**Registration chain:**

```mermaid
flowchart TD
    R1["include/Conversion/<br/>ConversionPasses.td<br/>def ExpandMathToArith"] --> R2["include/Conversion/<br/>ConversionPasses.h<br/>createExpandMathToArithPass()"]
    R1 --> R3["lib/Conversion/<br/>CMakeLists.txt<br/>add_subdirectory + link"]
    R4["lib/TaskflowDialect/<br/>TaskflowPasses.cpp<br/>insert into pipeline"] --> R3
```

### `strip-taskflow-task`: Remove Taskflow Containers

**File:** `lib/Conversion/TaskflowToNeura/StripTaskflowTaskPass.cpp`

**Motivation:** Stage 3 needs pure Neura dialect, but Stage 2 output wraps `neura.kernel` inside `taskflow.task`. This pass is the first step of Stage 3 — it lifts kernels to the parent function level.

**Algorithm:**

```mermaid
flowchart TD
    A["Scan for taskflow.task"] --> B["Find inner neura.kernel"]
    B --> C["Clone kernel + non-terminator<br/>helper ops before task"]
    C --> D["Map task block arguments<br/>to task operands<br/>(reads / writes / values)"]
    D --> E{"Handle returns"}
    E -->|"Read results<br/>(first N-2)"| F["Passthrough read inputs"]
    E -->|"Write result<br/>(last 1)"| G["Map to kernel results<br/>or fallback to write inputs"]
    G --> H["Erase taskflow.task"]
    F --> H
```

---

## Interpreter Implementation

**File:** `tools/neura-interpreter/neura-interpreter.cpp` (+2181 lines)

The interpreter executes Neura IR in two modes: **control-flow** (CF, sequential instruction dispatch) and **dataflow** (DF, dependency-graph-driven scheduling). The Python frontend tests invoke both.

```mermaid
flowchart TD
    subgraph Input["Input"]
        I1["Neura IR (.mlir)"]
    end

    subgraph Parse["Parsing Phase"]
        P1["MLIR Parse<br/>→ ModuleOp"] --> P2["Build DependencyGraph<br/>for each kernel"]
    end

    subgraph Dispatch["Execution Modes"]
        D_CF["Control-Flow Mode<br/>sequential instruction dispatch<br/>per SSA order"] --> Result
        D_DF["Dataflow Mode<br/>topological schedule<br/>from DependencyGraph"] --> Result
    end

    subgraph Handler["Op Handlers"]
        H1["Neura: kernel, load, store,<br/>addf, mulf, exp, rsqrt, etc."]
        H2["Taskflow: task, yield, counter<br/>(no-ops)"]
        H3["Math: tanh<br/>(delegated to sub-ops)"]
        H4["Generic: constant, select, cmp"]
    end

    I1 --> P1
    P1 --> P2
    P2 --> Dispatch
    Handler --> Dispatch

    Result["Output: store values<br/>+ comparison with golden"]
```

### Taskflow Op Handling

The interpreter must process three taskflow ops that appear in Stage 2 output:

| Op | Strategy | Rationale |
|----|----------|-----------|
| `taskflow.task` | No-op + frame push | Body is already expanded during collection; acts as frame push to enter task body |
| `taskflow.yield` | No-op | Terminator, skipped during execution |
| `taskflow.counter` | No-op | Wraps `neura.counter`; semantics handled by kernel counter ops |

### Dataflow Dependency Graph Enhancement

The `DependencyGraph::build()` was extended with a second phase: **memory dependency analysis**.

```mermaid
flowchart TD
    Phase1["Phase 1: SSA Dependencies<br/>Standard def-use edges"] --> Phase2["Phase 2: Memory Dependencies<br/>(★ new)"]

    subgraph Problem2["Why Memory Dependencies?"]
        P_A["neura.store_indexed<br/>No SSA result"] -.-> P_B["neura.load_indexed<br/>Same memref"]
        P_B -.-> P_C["No SSA edge between them!<br/>DF scheduler may reorder"]
    end

    subgraph Solution2["Solution"]
        S_A["Track per-memref<br/>last-access op table"] --> S_B["Insert RAW edges<br/>between consecutive<br/>load/store to same memref"]
        S_B --> S_C["Conservative but correct<br/>serialization"]
    end

    Phase2 --> Problem2
    Problem2 --> Solution2
```

**The problem:** In dataflow mode, kernel body ops are flattened into a single sequence. `neura.store_indexed` has no SSA result, so subsequent `neura.load_indexed` on the same memref lacks an SSA dependency edge, potentially causing read-write reordering.

**The fix:** Maintain a `memref_last_op` table during graph construction. For each load/store on the same memref, insert RAW (Read-After-Write), WAR (Write-After-Read), and WAW (Write-After-Write) dependency edges.

**Problem — no dependency edges, ops can reorder:**

```mermaid
flowchart TD
    subgraph Wrong["Problem: no memory deps"]
        direction TB
        A(["Op1: store → mem1[0]"])
        B(["Op2: store → mem1[1]"])
        C(["Op3: load ← mem1[0]"])
        D(["Op4: store → mem1[0]"])
    end
    Wrong --> Bug["Bug: Op3 may read stale data<br/>Bug: Op4 may overwrite before Op3 reads<br/>Bug: Op1 may reorder past Op4"]
```

**Solution — `memref_last_op` table + RAW / WAR / WAW edges:**

```mermaid
flowchart TD
    Step1(["Op1: store → mem1[0]"]) -->|"table empty, skip"| T1(["mem1[0] = Op1"])
    T1 --> Step2(["Op2: store → mem1[1]"])
    Step2 -->|"new index, skip"| T2(["mem1[0] = Op1<br/>mem1[1] = Op2"])
    T2 --> Step3(["Op3: load ← mem1[0]"])
    Step3 -->|"last = Op1 (store)<br/>→ insert RAW"| T3(["mem1[0] = Op3<br/>mem1[1] = Op2"])
    T3 --> Step4(["Op4: store → mem1[0]"])
    Step4 -->|"last = Op3 (load) → WAR<br/>last_store = Op1 → WAW"| T4(["mem1[0] = Op4<br/>mem1[1] = Op2"])
    T4 --> Result(["Op1 → Op3  (RAW)<br/>Op1 → Op4  (WAW)<br/>Op3 → Op4  (WAR)"])
```

**Final dependency graph — 4 ops, 3 edges, correct ordering:**

```mermaid
flowchart TD
    Op1(["Op1: store → mem1[0]"])
    Op2(["Op2: store → mem1[1]"])
    Op3(["Op3: load ← mem1[0]"])
    Op4(["Op4: store → mem1[0]"])

    Op1 -->|"RAW"| Op3
    Op1 -.->|"WAW"| Op4
    Op3 -->|"WAR"| Op4
```

### Simulated Memory

- `neura.load_indexed` / `neura.store_indexed`: use a `simulated_memory` hashmap (`unordered_map<string, float>`) keyed by `"memref_name.index"` for reads and writes
- `math.tanh`: treated as no-op in the handler — its expanded sub-ops (`math.exp`, `arith.*`) are handled individually

### `neura.kernel`: CGRA Offload Unit

`neura.kernel` is the fundamental **hardware computation unit** in Neura. It marks a region to be offloaded to a CGRA (Coarse-Grained Reconfigurable Architecture) accelerator.

```mermaid
flowchart LR
    subgraph CPU["CPU"]
        Alloc["memref.alloc"]
        Const["arith.constant"]
    end

    CPU -->|"inputs (memrefs, scalars)"| CGRA

    subgraph CGRA["CGRA — neura.kernel"]
        BB["bb0(%arg0)"]
        Ctr["neura.counter\n(lower=0, upper=4, step=1)"]
        Load(["load_indexed %arg0[i]"])
        Fmul(["fmul %v, %v"])
        Store(["store_indexed %r → %arg0[i]"])
        Yield["neura.yield"]

        BB --> Ctr
        Ctr -->|"index"| Load
        Ctr -.->|"predicate"| Load
        Load --> Fmul --> Store --> Yield
    end

    CGRA -->|"results"| Return["func.return"]
```

**IR structure:**

```mlir
neura.kernel inputs(%mem : memref<4xf32>) attributes {accelerator = "neura"} {
^bb0(%arg0: memref<4xf32>):   // block args map 1:1 to inputs
  %i = neura.counter ... -> !neura.data<i64, i1>
  %v = neura.load_indexed %arg0[%i] : memref<4xf32> -> f32
  %r = neura.fmul %v, %v : (f32, f32) -> f32
  neura.store_indexed %r -> %arg0[%i] : f32, memref<4xf32>
  neura.yield
}
```

| Component | Role |
|-----------|------|
| `inputs(...)` | Values captured from outside (memrefs, scalars) — must be explicit because kernel is `IsolatedFromAbove` |
| block args | 1:1 mapping to `inputs`; serve as local handles inside the kernel body |
| `neura.counter` | Hardware loop iterator: produces current index + `i1` predicate for gating |
| body ops | `load_indexed` / compute / `store_indexed` chain — pure dataflow inside the kernel |
| `neura.yield` | Terminator; optionally returns `iter_args_next` and `results` |

**Conceptually:** a `neura.kernel` is like a GPU kernel launch — it captures inputs from the CPU, runs an entire nested loop on the accelerator, and returns results. Multiple kernels in one function run **sequentially**, forming a pipeline of offloaded compute stages.

### Kernel-Level Scheduling

The interpreter supports two execution modes:

```mermaid
flowchart LR
    subgraph CF["Control Flow Mode"]
        C1["frame stack"] --> C2["walk ops linearly"]
        C2 --> C3["kernel? push frame"]
        C3 --> C4["yield? check counter"]
        C4 --> C5["overflow? pop frame"]
    end
    subgraph DF["Dataflow Mode"]
        D1["scan + group kernels"] --> D2["build DependencyGraph"]
        D2 --> D3["execute ready ops"]
        D3 --> D4["propagate to consumers"]
        D4 --> D5["advance counters"]
    end
```

#### Phase 1: Kernel Discovery

Scans the top-level `func.func` body, separating `neura.kernel` into groups and collecting non-kernel init ops (`arith.constant`, `memref.alloc`, etc.):

```mermaid
flowchart TD
    Scan["Scan func body ops"] --> IsKernel{"isa KernelOp?"}
    IsKernel -->|"yes"| Collect["collect body ops into KernelGroup\n+ extract counter values"]
    IsKernel -->|"no"| NonKernel["push to non_kernel_ops"]
    Collect --> Next["next op"]
    NonKernel --> Next
    Next --> IsKernel
```

#### Phase 2: Execute Non-Kernel Init Ops

Constants, `memref.alloc`, `memref.get_global` run once via `DependencyGraph` to initialise the environment before any kernel executes.

#### Phase 3: Kernel Scheduling Loop

**Inter-kernel: sequential.** Kernels run one after another to prevent cross-kernel read-write reordering.

**Intra-kernel: dataflow-driven.** Inside each kernel, ops execute when all their producers have finished:

```mermaid
flowchart TD
    Outer["for each KernelGroup (sequential)"] --> Map["map kernel block args to inputs"]
    Map --> Build["build DependencyGraph for kernel body ops"]
    Build --> Ready{"ready ops > 0?"}
    Ready -->|"yes"| Exec["execute all ready ops"]
    Exec --> Update["update dep graph\npropagate to consumers"]
    Update --> Ready
    Ready -->|"no"| Advance["advanceCountersNested()"]
    Advance --> HasMore{"more iterations?"}
    HasMore -->|"yes"| Reset["reset dep graph\nfor next iteration"]
    Reset --> Map
    HasMore -->|"no"| Outer
```

#### Counter Mechanism: Nested Loop Iteration

Each `neura.kernel` may contain **counter ops** that drive loop execution — the hardware equivalent of `for (i = 0; i < N; i += 1)`. A counter has:

- **bounds** (lower / upper / step) — define the iteration range
- **current index** — the loop variable, exposed as an SSA value for indexing (e.g. `memref[x][i]`)
- **predicate** — an `i1` flag: `true` while in bounds, `false` on overflow; downstream ops use this to gate execution in hardware

**Nested loops** use a counter hierarchy: **leaf** (innermost) → **relay** (middle) → **root** (outer). When inner counter overflows, it resets to `lower` and carries into the next outer counter — exactly like nested `for` loops in software.

```mermaid
flowchart LR
    subgraph Example["e.g. root=2, relay=2, leaf=2"]
        I0["(0,0,0)"] --> I1["(0,0,1)"] --> I2["leaf overflow → relay++"]
        I2 --> I3["(0,1,0)"] --> I4["(0,1,1)"] --> I5["leaf→relay→root overflow → done"]
    end
```

The carry-propagation algorithm:

```mermaid
flowchart TD
    Start["advance leaf counter"] --> LeafOver{"overflow?"}
    LeafOver -->|"no"| Done["next iteration ready"]
    LeafOver -->|"yes"| HasRelay{"has relay?"}
    HasRelay -->|"yes"| AdvRelay["advance relay + reset leaf"]
    HasRelay -->|"no"| HasRoot{"has root?"}
    AdvRelay --> RelayOver{"overflow?"}
    RelayOver -->|"no"| Done
    RelayOver -->|"yes"| HasRoot
    HasRoot -->|"yes"| AdvRoot["advance root + reset relay"]
    HasRoot -->|"no"| End["loop ends"]
    AdvRoot --> RootOver{"overflow?"}
    RootOver -->|"no"| Done
    RootOver -->|"yes"| End
```

**Summary:** kernels run sequentially; kernel-internal ops execute dataflow-style via `DependencyGraph` (SSA + memory deps); loop iterations are driven by hardware counter carry-propagation.

---

## Arithmetic Lowering

**File:** `lib/Conversion/ArithToNeura/ArithToNeuraPass.cpp`

To support complex models like GELU and LayerNorm, new arith/math → neura conversions were added:

| Pattern | Source Op | Target Op | Trigger Model |
|---------|-----------|-----------|---------------|
| `ArithCmpFToNeuraFCmp` | `arith.cmpf` | `neura.fcmp` | ReLU / GELU / LayerNorm |
| `MathExpToNeuraExp` | `math.exp` | `neura.exp` | GELU (tanh expansion) |
| `MathRsqrtToNeuraRsqrt` | `math.rsqrt` | `neura.rsqrt` | LayerNorm |
| `MathFPowIToNeuraMuls` | `math.fpowi` | `arith.mulf` chain | GELU (fallback) |
| `MathTanhToNeuraOps` | `math.tanh` | `exp + arith` chain | GELU tanh approx |

### Critical Fix: `arith.cmpf` Type Mismatch

```mermaid
flowchart LR
    subgraph Before["Before Fix"]
        B1["arith.cmpf"] --> B2["LeveragePredicatedValuePass<br/>wraps values into<br/>!neura.data&lt;f32, i1&gt;"]
        B2 --> B3["arith.cmpf expects f32<br/>receives !neura.data<br/>→ CRASH"]
    end

    subgraph After["After Fix"]
        A1["arith.cmpf"] --> A2["ArithCmpFToNeuraFCmp<br/>converts to neura.fcmp<br/>before LeveragePredicatedValue"]
        A2 --> A3["neura.fcmp natively handles<br/>!neura.data → OK"]
    end
```

`arith.cmpf` expected `f32` but received `!neura.data<f32, i1>` after the `LeveragePredicatedValuePass` performed type wrapping. Converting to `neura.fcmp` ahead of time ensures the type system remains consistent.

---

## Testing

### End-to-End Numerical Tests

**File:** `test/Conversion/python2neura/model_tests/test_models.py`

```mermaid
flowchart TD
    A["PyTorch Model (.py)"] --> B["Stage 0: Export<br/>→ Linalg MLIR"]
    B --> C["Generate input<br/>(Goldfish hash seed)"]
    C --> D["Run PyTorch model<br/>→ golden reference"]
    B --> E["Pipeline Stages 1-3<br/>→ Neura + Codegen"]
    E --> F["neura-interpreter<br/>CF mode + DF mode"]
    F --> G["Parse store outputs"]
    D --> H["Element-wise comparison<br/>max_abs_err, max_rel_err,<br/>mean_abs_err"]
    G --> H
    H --> I["PASS / FAIL"]
```

**8 test models:**

| Model | Input Shape | Key Ops |
|-------|-------------|---------|
| `simple_matmul` | [4, 8] | matmul |
| `residual_block` | [4, 8] | matmul + add + relu |
| `residual_block_norelu` | [4, 8] | matmul + add |
| `two_layer_mlp` | [2, 8] | two-layer matmul + relu |
| `two_layer_mlp_norelu` | [2, 8] | two-layer matmul |
| `conv2d_relu_pool` | [1, 9] | conv2d + relu + max_pool2d |
| `transformer_attention` | [4, 8] | multi-head attention |
| `gelu_layernorm` | [4, 8] | GELU + LayerNorm |

**Run:**

```bash
conda activate neura-torch

# Control-flow mode
python test/Conversion/python2neura/model_tests/test_models.py

# Dataflow mode
python test/Conversion/python2neura/model_tests/test_models.py --dataflow
```

**Results (all 8/8 pass):**

```
============================================================
  E2E TEST SUMMARY
============================================================
  simple_matmul                  OK PASS
  residual_block                 OK PASS
  residual_block_norelu          OK PASS
  two_layer_mlp                  OK PASS
  two_layer_mlp_norelu           OK PASS
  conv2d_relu_pool               OK PASS
  transformer_attention          OK PASS
  gelu_layernorm                 OK PASS

  8/8 models passed
```

Precision (simple_matmul example): `max_abs_err=4.66e-09 max_rel_err=2.64e-06 mean_abs_err=7.90e-10`

### Lit Static Regression Tests

Pre-generated `.mlir` files with embedded `RUN` + `CHECK` directives:

```bash
llvm-lit test/Conversion/python2neura/model_tests/generated/
```

To regenerate:

```bash
python test/Conversion/python2neura/model_tests/generate_mlir.py \
    --output-dir test/Conversion/python2neura/model_tests/generated --lit
```

---

## Key Fixes Summary

| Problem | Root Cause | Fix |
|---------|------------|-----|
| torch-mlir API break | `compile()` → `export_and_import()` | Adopt new API + `OutputType` enum |
| GELU `x^3` not expandable | Constant exponent hidden after promotion | `expand-math-to-arith` pass before affine lowering |
| `arith.cmpf` type mismatch | Predicated value wrapping conflicts with f32 input | `ArithCmpFToNeuraFCmp` converts to `neura.fcmp` first |
| DF mode read-write reorder | `store_indexed` has no SSA result | Per-memref memory dependency edges in DependencyGraph |
| Interpreter "Unhandled op" | Taskflow ops not handled | No-op branches for `taskflow.task/yield/counter` |

---

## Dependencies

- Python 3.11+
- `torch` (≥ 2.3.0)
- `torch-mlir` (20260531.828)
- `numpy`
- Project binaries: `mlir-neura-opt`, `neura-interpreter`, `neura-compiler`

Install via conda:

```bash
conda env create -f tools/neura-py-frontend/environment.yml
```

**Build integration:**

- CMake target `neura-py-frontend` copies scripts to `build/tools/neura-py-frontend/`
- New passes registered in `include/Conversion/ConversionPasses.td` + `.h`
- Pass pipeline integration in `lib/TaskflowDialect/TaskflowPasses.cpp`
- Interpreter updates in `tools/neura-interpreter/neura-interpreter.cpp`

---

## File Changes (vs main)

**44 files, +5726/-293 lines:**

| Category | Files | Type |
|----------|-------|------|
| Frontend scripts | `neura_pipeline.py`, `neura-py-frontend.sh`, `CMakeLists.txt` | Added |
| Docs & config | `README.md`, `environment.yml` | Added |
| New passes | `ExpandMathToArith/`, `StripTaskflowTaskPass.cpp` | Added |
| Interpreter | `neura-interpreter.cpp` | Modified |
| Arithmetic conversion | `ArithToNeuraPass.cpp` | Modified |
| Pass registration | `ConversionPasses.h/.td`, `NeuraOps.td`, `CMakeLists.txt` | Modified |
| Test models | `models/*.py` (8), `generated/*.mlir` (8) | Added |
| Test scripts | `test_models.py`, `generate_mlir.py` | Added |
