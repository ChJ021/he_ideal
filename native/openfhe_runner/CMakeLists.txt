cmake_minimum_required(VERSION 3.21)
project(hetune_openfhe_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenFHE CONFIG REQUIRED)

add_executable(hetune_openfhe_runner src/main.cpp)
target_include_directories(
    hetune_openfhe_runner
    PRIVATE
        ${OpenFHE_INCLUDE}
        ${OpenFHE_INCLUDE}/core
        ${OpenFHE_INCLUDE}/pke
        ${OpenFHE_INCLUDE}/binfhe
        ${OpenFHE_INCLUDE}/cereal
)
target_link_libraries(hetune_openfhe_runner PRIVATE OPENFHEcore OPENFHEpke)

separate_arguments(OPENFHE_CXX_FLAGS_LIST UNIX_COMMAND "${OpenFHE_CXX_FLAGS}")
target_compile_options(hetune_openfhe_runner PRIVATE ${OPENFHE_CXX_FLAGS_LIST})
