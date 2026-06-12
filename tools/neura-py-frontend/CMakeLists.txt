# neura-py-frontend — Python frontend lowering pipeline
#
# This does not build a C++ binary. Instead it installs the Python pipeline
# script and a convenience shell wrapper alongside the built tools.

set(PY_FRONTEND_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/neura_pipeline.py")
set(PY_FRONTEND_WRAPPER "${CMAKE_CURRENT_SOURCE_DIR}/neura-py-frontend.sh")

# Copy the Python script to the build output directory so it sits alongside
# neura-compiler / mlir-neura-opt binaries.
add_custom_target(neura-py-frontend ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PY_FRONTEND_SCRIPT}"
        "${CMAKE_CURRENT_BINARY_DIR}/neura_pipeline.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PY_FRONTEND_WRAPPER}"
        "${CMAKE_CURRENT_BINARY_DIR}/neura-py-frontend"
    COMMAND chmod +x "${CMAKE_CURRENT_BINARY_DIR}/neura_pipeline.py"
    COMMAND chmod +x "${CMAKE_CURRENT_BINARY_DIR}/neura-py-frontend"
    COMMENT "Installing neura-py-frontend Python pipeline scripts"
)
