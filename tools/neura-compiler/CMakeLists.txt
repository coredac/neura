add_executable(neura-compiler neura-compiler.cpp)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRNeuraTransforms
        MLIRConversion
        MLIRNeura
        MLIRTransforms
        MLIROptLib
        MLIRPass
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRFuncAllExtensions
        MLIRTensorAllExtensions
        )

target_link_libraries(neura-compiler PRIVATE ${LIBS})