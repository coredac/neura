add_mlir_public_c_api_library(MLIRNeuraCAPI
    Neura.cpp

    LINK_LIBS PUBLIC
    MLIRNeura
)