add_executable(neura-interpreter neura-interpreter.cpp)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRNeuraTransforms
        MLIRNeuraConversion
        MLIRNeura
        MLIRTransforms
        MLIROptLib
        MLIRPass
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRInferTypeOpInterface
        )

target_link_libraries(neura-interpreter PRIVATE ${LIBS})

mlir_check_all_link_libraries(neura-interpreter)