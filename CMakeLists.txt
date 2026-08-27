cmake_minimum_required(VERSION 3.13.4)
project(NeuraMLIRProject VERSION 0.1.0)
include(GNUInstallDirs)

# Avoid cmake generate unknown warnings
cmake_policy(SET CMP0116 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_compile_options(-g)

# set(MLIR_DIR /home/lucas/llvm-project/build/lib/cmake/mlir)
# set(LLVM_DIR /home/lucas/llvm-project/build/lib/cmake/llvm)
# set(MLIR_SRC_DIR /home/lucas/llvm-project/mlir)
# set(MLIR_BINARY_DIR /home/lucas/llvm-project/build)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(BEFORE ${PROJECT_SOURCE_DIR}/include)
include_directories(BEFORE ${PROJECT_BINARY_DIR}/include)

# Package the analytical mapper beside build/install artifacts before the
# mapping library is configured, so its default path is independent of cwd.
set(NEURA_CPSAT_SCRIPT_SOURCE
    ${CMAKE_SOURCE_DIR}/test/cost-model/exact_mapper_cpsat.py)
set(NEURA_CPSAT_SCRIPT_BUILD
    ${CMAKE_BINARY_DIR}/share/neura/exact_mapper_cpsat.py)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/share/neura)
configure_file(${NEURA_CPSAT_SCRIPT_SOURCE} ${NEURA_CPSAT_SCRIPT_BUILD}
               COPYONLY)
install(FILES ${NEURA_CPSAT_SCRIPT_SOURCE}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/neura)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# ===--------------------------------------------------------------------===#
# LIT test setup
# ===--------------------------------------------------------------------===#

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
# Tools built by this project or LLVM
set(MLIR_NEURA_OPT ${CMAKE_BINARY_DIR}/tools/mlir-neura-opt/mlir-neura-opt)
set(NEURA_INTERPRETER ${CMAKE_BINARY_DIR}/tools/neura-interpreter/neura-interpreter)
set(NEURA_COMPILER ${CMAKE_BINARY_DIR}/tools/neura-compiler/neura-compiler)
set(FILECHECK ${LLVM_TOOLS_BINARY_DIR}/FileCheck)
set(MLIR_OPT ${LLVM_TOOLS_BINARY_DIR}/mlir-opt)
set(MLIR_TRANSLATE ${LLVM_TOOLS_BINARY_DIR}/mlir-translate)
set(LLC ${LLVM_TOOLS_BINARY_DIR}/llc)
set(CLANG ${LLVM_TOOLS_BINARY_DIR}/clang)
set(NEURA_PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE FILEPATH
    "Python interpreter used by Neura's CP-SAT e2e tests")

# Configures lit.cfg from lit.cfg.in
configure_file(
  ${CMAKE_SOURCE_DIR}/test/lit.cfg.in
  #  ${CMAKE_BINARY_DIR}/test/lit.cfg
  ${CMAKE_SOURCE_DIR}/test/lit.cfg
  @ONLY
)

# Adds a custom target for running lit tests
add_custom_target(check-neura
  COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_BINARY_DIR}/test
  DEPENDS mlir-neura-opt
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running Sora Dialect Tests with lit"
)
endif()
