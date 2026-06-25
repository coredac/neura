add_mlir_library(MLIRTaskflowThroughputGuidedTaskOrchestration
    ResourceAssignmentState.cpp
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
