get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRTaskflowTransforms
    AllocateCgraToTaskPass.cpp
    ConstructHyperblockFromTaskPass.cpp
    ClassifyCountersPass.cpp
    MapTaskOnCgraPass.cpp
    FuseTaskPass.cpp

    DEPENDS
    MLIRTaskflowTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTaskflow
    MLIRTaskflowAllocation
    MLIRNeura
    MLIRNeuraTransforms
    MLIRConversion
    ${dialect_libs}
    LLVMSupport
)

add_subdirectory(Optimizations)