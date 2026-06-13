# Neura Python Frontend

The Python frontend compiles PyTorch models through the Neura CGRA dataflow
lowering pipeline, from `torch.nn.Module` down to Neura IR and (optionally)
CGRA JSON code generation.

## Files

| File | Purpose |
|------|---------|
| `neura_pipeline.py` | Python API + CLI for the full lowering pipeline |
| `neura-py-frontend.sh` | Convenience shell wrapper (forwards args to `neura_pipeline.py`) |
| `environment.yml` | Conda environment with torch + torch-mlir dependencies |
| `CMakeLists.txt` | Installs scripts into the build output directory |

## Environment Setup

```bash
# 1. Create the conda environment
conda env create -f tools/neura-py-frontend/environment.yml
conda activate neura-torch

# 2. Build the project (produces mlir-neura-opt, neura-interpreter, neura-compiler)
#    Point MLIR_DIR and LLVM_DIR to your llvm-project build.
mkdir -p build && cd build
cmake .. -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
         -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm
make -j$(nproc) mlir-neura-opt neura-interpreter neura-compiler neura-py-frontend
```

After building, the binaries live under `build/tools/`:

- `build/tools/mlir-neura-opt/mlir-neura-opt` — MLIR optimization driver
- `build/tools/neura-interpreter/neura-interpreter` — Neura IR interpreter
- `build/tools/neura-compiler/neura-compiler` — Compiler frontend

The Python pipeline script auto-discovers these binaries relative to its own
source location (see `_find_tool()` in `neura_pipeline.py`).

## Pipeline Architecture

```
PyTorch nn.Module  (.py)
    │  torch-mlir (export_and_import → OutputType.LINALG_ON_TENSORS)
    ▼
Linalg-on-Tensors  (.mlir)
    │  Stage 1: --linalg-to-affine-conversion
    ▼
Affine  (.mlir)
    │  Stage 2: --taskflow-conversion
    ▼
Neura  (.mlir)
    │  Stage 3: --neura-conversion
    ▼
Optimised Neura + Codegen  (.mlir / .json)
```

There is also an all-in-one pipeline `--python-to-neura` that folds all three
stages into a single `mlir-neura-opt` invocation.

All passes are registered in:

- `lib/TaskflowDialect/TaskflowPasses.cpp` — `linalg-to-affine-conversion`, `taskflow-conversion`
- `lib/NeuraDialect/NeuraPasses.cpp` — `neura-conversion`, `python-to-neura`

## CLI Usage

```bash
# From a PyTorch model file
python tools/neura-py-frontend/neura_pipeline.py \
    --model path/to/model.py \
    --example-shape 1 3 224 224 \
    --output output.json

# From an existing Linalg-on-Tensors .mlir file
python tools/neura-py-frontend/neura_pipeline.py \
    --input linalg_ir.mlir \
    --output output.json

# Stop after Stage 2 (output is Neura dialect, before codegen)
python tools/neura-py-frontend/neura_pipeline.py \
    --input linalg_ir.mlir \
    --output output.mlir \
    --stop-after taskflow

# With architecture specification
python tools/neura-py-frontend/neura_pipeline.py \
    --model path/to/model.py \
    --example-shape 4 8 \
    --arch-spec test/arch_spec/architecture.yaml \
    --output output.json
```

Or via the convenience wrapper (same arguments, must build `neura-py-frontend` target first):

```bash
build/tools/neura-py-frontend/neura-py-frontend --model model.py --output out.json
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Path to `.py` file exposing a `model` attribute |
| `--input`, `-i` | Path to an existing Linalg-on-Tensors `.mlir` file |
| `--output`, `-o` | Output path (default: `output.json`) |
| `--stop-after` | Stop after stage: `affine` (after Stage 1), `taskflow` (after Stage 2, Neura dialect before codegen), or `neura` (after Stage 3, same as default — fully lowered with codegen) |
| `--keep-intermediate` | Preserve intermediate `.mlir` files on disk |
| `--func-name` | Function name for torch-mlir export (default: `forward`) |
| `--example-shape` | Example input shape, e.g. `1 3 224 224` |
| `--arch-spec` | Path to CGRA architecture specification YAML |

`--model` and `--input` are mutually exclusive.

## Programmatic API

```python
from neura_pipeline import (
    export_pytorch_to_linalg,   # PyTorch .py → Linalg-on-Tensors .mlir
    lower_linalg_to_affine,     # Linalg → Affine (--linalg-to-affine-conversion)
    lower_affine_to_neura,      # Affine → Neura  (--taskflow-conversion)
    lower_neura_to_codegen,     # Neura → Codegen  (--neura-conversion)
    full_pipeline,              # Linalg → Affine → Neura → Codegen
)

# Step by step
export_pytorch_to_linalg("model.py", "linalg.mlir", example_shape=[1, 3, 224, 224])
lower_linalg_to_affine("linalg.mlir", "affine.mlir")
lower_affine_to_neura("affine.mlir", "neura.mlir")
lower_neura_to_codegen("neura.mlir", "output.json")

# Or in one call
full_pipeline(
    input_mlir="linalg.mlir",
    output_file="output.json",
    arch_spec=None,          # optional architecture YAML
    stop_after=None,          # or "affine", "taskflow", "neura"
    keep_intermediate=False,
)
```

All functions return `True` on success, `False` on failure.

## Model File Convention

Each model `.py` file must expose a `torch.nn.Module` instance named `model`.

Example (`models/simple_matmul.py`):

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

The 8 built-in test models under `test/Conversion/python2neura/model_tests/models/`:

- `simple_matmul.py`
- `residual_block.py`
- `residual_block_norelu.py`
- `two_layer_mlp.py`
- `two_layer_mlp_norelu.py`
- `conv2d_relu_pool.py`
- `transformer_attention.py`
- `gelu_layernorm.py`

## Stage Details

### Stage 0: PyTorch → Linalg-on-Tensors

Performed by `export_and_import(model, example_input, output_type=OutputType.LINALG_ON_TENSORS)`
from `torch-mlir`.  This step runs inside Python and does not invoke `mlir-neura-opt`.

### Stage 1: `--linalg-to-affine-conversion`

Registered in `lib/TaskflowDialect/TaskflowPasses.cpp`. The pipeline:

1. `linalg-generalize-named-ops` — generalize named ops to generic form
2. `eliminate-empty-tensors`
3. `canonicalize`
4. `one-shot-bufferize` — tensor → memref
5. `convert-linalg-to-affine-loops`
6. `fold-memref-alias-ops`
7. `fold-affine-subview`, `convert-copy-to-affine-loops`
8. `affine-loop-normalize`, `simplify-affine-structures`
9. `canonicalize`
10. `expand-math-to-arith` — expand math ops before taskflow conversion

### Stage 2: `--taskflow-conversion`

Registered in `lib/TaskflowDialect/TaskflowPasses.cpp`:

1. `convert-affine-to-taskflow` — Affine → Taskflow
2. `construct-hyperblock-from-task`
3. `classify-task-and-counter`
4. `convert-taskflow-to-neura` — Taskflow → Neura

### Stage 3: `--neura-conversion`

Registered in `lib/NeuraDialect/NeuraPasses.cpp`:

1. `strip-taskflow-task` — remove taskflow.task wrappers
2. `assign-accelerator`
3. `lower-affine-to-neura`, `lower-arith-to-neura`, `lower-memref-to-neura`, `lower-builtin-to-neura`, `lower-llvm-to-neura`
4. `canonicalize-return`, `canonicalize-cast`
5. `promote-input-arg-to-const`, `fold-constant`, `canonicalize-live-in`
6. `leverage-predicated-value`
7. `transform-ctrl-to-data-flow`, `fold-constant`
8. `fuse-pattern`, `insert-data-mov`
9. `map-to-accelerator`, `generate-code`

### Bonus: `--python-to-neura`

An all-in-one pipeline merging the three stages above. Invoked directly on
Linalg-on-Tensors IR, it produces the final codegen output in a single pass.

## Testing

Tests live under `test/Conversion/python2neura/model_tests/`.

### Lit tests — static MLIR regression

Pre-generated `.mlir` files with embedded `RUN` and `CHECK` directives:

```bash
llvm-lit test/Conversion/python2neura/model_tests/generated/
```

To regenerate lit files from the current models and toolchain:

```bash
python test/Conversion/python2neura/model_tests/generate_mlir.py \
    --output-dir test/Conversion/python2neura/model_tests/generated --lit
```

### End-to-end numerical tests

Compiles each model through the full pipeline, runs the interpreter, and
compares output element-by-element against a PyTorch golden reference:

```bash
# Make sure the conda environment is active
conda activate neura-torch

# Control-flow mode (default)
python test/Conversion/python2neura/model_tests/test_models.py

# Dataflow mode
python test/Conversion/python2neura/model_tests/test_models.py --dataflow
```

Example output (control-flow mode):

```
============================================================
  simple_matmul  (shape=[4, 8])
============================================================
  [Stage 0] PyTorch -> Linalg ...
  Linalg->Affine OK
  Affine->Neura OK
  Neura->Dataflow OK
  [Stage 4] Interpreter (control-flow) ...
    m0: shape=[4, 16] stores=64 non_zero=True
  PASS m0 shape=[4, 16] | max_abs_err=4.66e-09 max_rel_err=2.64e-06 mean_abs_err=7.90e-10 (non-zero)
...
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

Example output (dataflow mode):

```
============================================================
  simple_matmul  (shape=[4, 8])
============================================================
  [Stage 0] PyTorch -> Linalg ...
  Linalg->Affine OK
  Affine->Neura OK
  Neura->Dataflow OK
  [Stage 4] Interpreter (dataflow) ...
    m0: shape=[4, 16] stores=64 non_zero=True
  PASS m0 shape=[4, 16] | max_abs_err=4.66e-09 max_rel_err=2.64e-06 mean_abs_err=7.90e-10 (non-zero)
...
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

## Dependencies

- Python 3.11+
- `torch` (>= 2.3.0)
- `torch-mlir`
- `numpy`
- Project binaries: `mlir-neura-opt`, `neura-interpreter`, `neura-compiler`

Install via the provided conda environment:

```bash
conda env create -f tools/neura-py-frontend/environment.yml
```
