add_mlir_library(MLIRTaskflowThroughputGuidedTaskOrchestration
    ThroughputGuidedTaskOrchestration.cpp

    DEPENDS
    MLIRTaskflowTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRTaskflow
    MLIRTaskflowOrchestrationUtil
    LLVMSupport
)
