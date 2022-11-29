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
add_executable(ProtonReconstruction macros/punch_through/proton_reconstruction.cc ${sources} ${headers} )
add_executable(QuasifreeReconstruction macros/quasifree_reconstruction/quasifree_reconstruction.cc ${sources} ${headers} )
add_executable(QuasifreeClassification macros/quasifree_reconstruction/quasifree_classification.cc ${sources} ${headers} )
add_executable(QuasifreeExcitationEnergy macros/quasifree_reconstruction/quasifree_exc_energy.cc ${sources} ${headers} )
add_executable(DeltaFission macros/delta_fission/function_regression_eofx.cc ${sources} ${headers} )



target_link_libraries(SoKAI ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(FunctionRegression ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(ProtonReconstruction ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeReconstruction ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeClassification ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeExcitationEnergy ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(DeltaFission ${ROOT_LIBRARIES} glog::glog)
