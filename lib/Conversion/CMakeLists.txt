get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_subdirectory(ArithToNeura)
add_subdirectory(AffineToNeura)
add_subdirectory(LlvmToNeura)
add_subdirectory(MemRefToNeura)
add_subdirectory(BuiltinToNeura)
add_subdirectory(TorchToNeura)

add_library(MLIRNeuraConversion INTERFACE)

add_dependencies(MLIRNeuraConversion MLIRNeuraConversionIncGen)

target_link_libraries(MLIRNeuraConversion INTERFACE
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRNeura
  MLIRNeuraArithToNeuraPass
  MLIRNeuraAffineToNeuraPass
  MLIRNeuraLlvmToNeuraPass
  MLIRNeuraMemRefToNeuraPass
  MLIRNeuraBuiltinToNeuraPass
  MLIRNeuraTorchToNeuraPass
  ${dialect_libs}
)