project(SoKAI)

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


include_directories(${PROJECT_SOURCE_DIR}/headers)

file(GLOB sources ${PROJECT_SOURCE_DIR}/sources/*.cxx)
file(GLOB headers ${PROJECT_SOURCE_DIR}/headers/*.h)

add_executable(SoKAI SoKAI.cc ${sources} ${headers})
