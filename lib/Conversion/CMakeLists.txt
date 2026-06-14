get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_subdirectory(ExpandMathToArith)
add_subdirectory(ArithToNeura)
add_subdirectory(AffineToNeura)
add_subdirectory(LlvmToNeura)
add_subdirectory(MemRefToNeura)
add_subdirectory(BuiltinToNeura)
add_subdirectory(AffineToTaskflow)
add_subdirectory(TaskflowToNeura)

add_library(MLIRConversion INTERFACE)

add_dependencies(MLIRConversion MLIRConversionIncGen)

target_link_libraries(MLIRConversion INTERFACE
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRNeura
  MLIRExpandMathToArithPass
  MLIRNeuraArithToNeuraPass
  MLIRNeuraAffineToNeuraPass
  MLIRNeuraLlvmToNeuraPass
  MLIRNeuraMemRefToNeuraPass
  MLIRNeuraBuiltinToNeuraPass
  MLIRAffineToTaskflowPass
  MLIRTaskflowToNeuraPass
  ${dialect_libs}
)