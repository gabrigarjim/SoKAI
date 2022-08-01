project(SoKAI)

set (CMAKE_CXX_STANDARD 11)

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
find_package (glog 0.5.0 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/headers)

file(GLOB sources ${PROJECT_SOURCE_DIR}/sources/*.cxx)
file(GLOB headers ${PROJECT_SOURCE_DIR}/headers/*.h)


find_package(ROOT)

EXECUTE_PROCESS(COMMAND root-config --cflags OUTPUT_VARIABLE ROOT_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ROOT_CXX_FLAGS}")

EXECUTE_PROCESS(COMMAND root-config --libs  OUTPUT_VARIABLE ROOT_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)

EXECUTE_PROCESS(COMMAND root-config --libs OUTPUT_VARIABLE ROOT_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_EXE_LINKER_FLAGS ${ROOT_LD_FLAGS})


add_executable(SoKAI macros/examples/classification/SoKAI.cc ${sources} ${headers} )
add_executable(FunctionRegression macros/examples/regression/function_regression.cc ${sources} ${headers} )

target_link_libraries(SoKAI ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(FunctionRegression ${ROOT_LIBRARIES} glog::glog)
