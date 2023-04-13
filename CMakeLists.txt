project(SoKAI)

set (CMAKE_CXX_STANDARD 17)

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
add_executable(QuasifreeReconstruction macros/quasifree_reconstruction/quasifree_reconstruction.cc ${sources} ${headers} )
add_executable(QuasifreeReconstructionSmaller macros/quasifree_reconstruction/quasifree_reconstruction_smaller.cc ${sources} ${headers} )
add_executable(QuasifreeClassification macros/quasifree_reconstruction/quasifree_classification.cc ${sources} ${headers} )
add_executable(QuasifreeClassificationBigger macros/quasifree_reconstruction/quasifree_classification_bigger.cc ${sources} ${headers} )
add_executable(QuasifreeClassificationSmaller macros/quasifree_reconstruction/quasifree_classification_smaller.cc ${sources} ${headers} )
add_executable(QuasifreeClassificationCheck macros/quasifree_reconstruction/quasifree_classification_check.cc ${sources} ${headers} )
add_executable(QuasifreeReconstructionCheck macros/quasifree_reconstruction/quasifree_reconstruction_check.cc ${sources} ${headers} )
add_executable(QuasifreeExpReconstruction macros/quasifree_reconstruction/exp_data_full_reconstruction.cc ${sources} ${headers} )
add_executable(KnockoutReconstruction macros/knockout_reconstruction/knockout_reconstruction.cc ${sources} ${headers} )
add_executable(KnockoutReconstructionCheck macros/knockout_reconstruction/knockout_reconstruction_check.cc ${sources} ${headers} )
add_executable(KnockoutClassification macros/knockout_reconstruction/knockout_classification.cc ${sources} ${headers} )
add_executable(KnockoutExpReconstruction macros/knockout_reconstruction/exp_data_full_reconstruction.cc ${sources} ${headers} )



target_link_libraries(SoKAI ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(FunctionRegression ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeReconstruction ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeReconstructionSmaller ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeClassification ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeClassificationCheck ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeReconstructionCheck ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeClassificationBigger ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeClassificationSmaller ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(QuasifreeExpReconstruction ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(KnockoutClassification ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(KnockoutReconstruction ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(KnockoutReconstructionCheck ${ROOT_LIBRARIES} glog::glog)
target_link_libraries(KnockoutExpReconstruction ${ROOT_LIBRARIES} glog::glog)




set_target_properties(FunctionRegression SoKAI
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties( QuasifreeReconstruction QuasifreeReconstructionSmaller QuasifreeClassification QuasifreeClassificationCheck QuasifreeReconstructionCheck QuasifreeClassificationBigger  QuasifreeClassificationSmaller
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/QuasifreeBinaries"
)

set_target_properties(QuasifreeExpReconstruction
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ExpBinaries"
)

set_target_properties(KnockoutReconstruction KnockoutClassification KnockoutReconstructionCheck KnockoutExpReconstruction
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/KnockoutBinaries"
)

set_target_properties(CrossSectionPrediction
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/CrossSectionsBinaries"
)
