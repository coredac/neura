# Neura -- a dataflow dialect.

## Build LLVM and Neura

Neura is pinned to LLVM commit `6146a88f60492b520a36f8f8f3231e15f3cc6082`.

### Prerequisites

Use Python 3.11.16 and install the native binding dependencies into the same
environment that will configure LLVM and Neura:

```sh
python -m pip install pybind11==2.13.6 nanobind==2.15.0
```

The system build dependencies are CMake, Ninja, Clang, LLD, and optionally
ccache.

### Build LLVM/MLIR

From the `llvm-project` repository:

```sh
git checkout 6146a88f60492b520a36f8f8f3231e15f3cc6082

cmake -G Ninja -S llvm -B build \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-std=c++17 -frtti" \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_INSTALL_AGGREGATE_OBJECTS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)"

cmake --build build
```

`MLIR_ENABLE_BINDINGS_PYTHON=ON` is required because Neura builds its Python
package from the upstream MLIR Python bindings.

### Build Neura

Neura is an out-of-tree MLIR project. Set `LLVM_BUILD_DIR` to the LLVM build
directory; Neura derives the LLVM source and CMake package paths from it.

```sh
cd /path/to/dataflow
export LLVM_BUILD_DIR=/path/to/llvm-project/build

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)"

cmake --build build
cmake --build build --target NeuraPythonModules
```

The generated package is placed in
`build/python_packages/neura_core/neura_mlir`.

### Run tests

```sh
# Focused Python operation-binding test.
$LLVM_BUILD_DIR/bin/llvm-lit -v test/python/neura_binding.py

# Complete Neura test suite.
$LLVM_BUILD_DIR/bin/llvm-lit -v test
```

Some frontend tests additionally require PyTorch and a compatible torch-mlir
wheel; the pinned CI setup is in `.github/workflows/main.yml`.

Sync `test/e2e` outputs into Zeonica_Testbench (submodule)
--------------------------------------------------------
This repo vendors [`sarchlab/Zeonica_Testbench`](https://github.com/sarchlab/Zeonica_Testbench.git) as a git submodule at `test/benchmark/Zeonica_Testbench`.

After running e2e compilation/tests, you can sync the generated artifacts into the testbench repo:
```sh
$ ./tools/sync_e2e_outputs_to_zeonica_testbench.sh
```

Mapping (per kernel `K`):
- `test/e2e/K/tmp-generated-dfg.{dot,yaml}` -> `test/benchmark/Zeonica_Testbench/kernel/K/K-dfg.{dot,yaml}`
- `test/e2e/K/tmp-generated-instructions.{asm,yaml}` -> `test/benchmark/Zeonica_Testbench/kernel/K/K-instructions.{asm,yaml}`

Contributing
--------------------------------------------------------
Please refer to the [Contributing Guide](https://github.com/coredac/dataflow?tab=contributing-ov-file#contributing-to-neura) for code style, formatting, and contribution workflow.
