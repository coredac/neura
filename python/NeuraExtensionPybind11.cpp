#include "Neura-C/Neura.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace mlir::python::adaptors;

PYBIND11_MODULE(_NeuraExtensionPybind11, module) {
  auto neura_module = module.def_submodule("neura");

  neura_module.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__neura__();

        mlirDialectHandleRegisterDialect(handle, context);

        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}