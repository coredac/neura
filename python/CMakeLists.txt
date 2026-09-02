include(AddMLIRPython)

# Build a relocatable private MLIR Python package named ``neura_mlir``.
add_compile_definitions(
  "MLIR_PYTHON_PACKAGE_PREFIX=neura_mlir."
)

# Neura-owned Python sources.
declare_mlir_python_sources(NeuraPythonSources)

# Generate Python OpView classes from the existing Neura ODS definitions.
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NeuraPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/NeuraOps.td
  SOURCES
    dialects/neura.py
  DIALECT_NAME neura
)

# Native extension used to register the Neura dialect.
declare_mlir_python_extension(
  NeuraPythonSources.PybindExtension

  MODULE_NAME _NeuraExtensionPybind11
  ADD_TO_PARENT NeuraPythonSources

  SOURCES
    NeuraExtensionPybind11.cpp

  EMBED_CAPI_LINK_LIBS
    MLIRNeuraCAPI

  PYTHON_BINDINGS_LIBRARY pybind11
)

set(
  NEURA_PYTHON_PACKAGE_DIR
  "${NEURA_PYTHON_PACKAGE_ROOT}/neura_mlir"
)

# Build one C API shared library used by the core and Neura extensions.
add_mlir_python_common_capi_library(NeuraPythonCAPI
  INSTALL_COMPONENT NeuraPythonModules

  INSTALL_DESTINATION
    python_packages/neura_core/neura_mlir/_mlir_libs

  OUTPUT_DIRECTORY
    "${NEURA_PYTHON_PACKAGE_DIR}/_mlir_libs"

  RELATIVE_INSTALL_ROOT "../../../.."

  DECLARED_SOURCES
    MLIRPythonSources.Core
    MLIRPythonExtension.RegisterEverything
    NeuraPythonSources
)

# Assemble the complete Python package.
add_mlir_python_modules(NeuraPythonModules
  ROOT_PREFIX
    "${NEURA_PYTHON_PACKAGE_DIR}"

  INSTALL_PREFIX
    "python_packages/neura_core/neura_mlir"

  DECLARED_SOURCES
    MLIRPythonSources.Core
    MLIRPythonExtension.RegisterEverything

    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.affine
    MLIRPythonSources.Dialects.arith
    MLIRPythonSources.Dialects.cf
    MLIRPythonSources.Dialects.func
    MLIRPythonSources.Dialects.linalg
    MLIRPythonSources.Dialects.math
    MLIRPythonSources.Dialects.memref
    MLIRPythonSources.Dialects.scf
    MLIRPythonSources.Dialects.tensor
    MLIRPythonSources.Dialects.vector

    NeuraPythonSources

  COMMON_CAPI_LINK_LIBS
    NeuraPythonCAPI
)