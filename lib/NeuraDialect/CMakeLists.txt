# Add the dialect library
add_mlir_dialect_library(MLIRNeura
        Neura.cpp
        NeuraOps.cpp
        NeuraTypes.cpp
        NeuraPasses.cpp
        Mapping/mapping_util.cpp
        Mapping/MappingState.cpp
        Mapping/Mapping.cpp
        Mapping/HeuristicMapping/HeuristicMapping.cpp
        Architecture/Architecture.cpp
        Transforms/GraphMining/GraMi.cpp
        Util/ArchParser.cpp
        Util/ParserUtils.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NeuraDialect

        DEPENDS
        MLIRNeuraTransformsIncGen
        MLIRNeuraConversionIncGen
        
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRInferTypeOpInterface
        )

add_subdirectory(Transforms)