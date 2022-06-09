cmake_minimum_required(VERSION 3.22)
project(cpp)
set(CMAKE_CXX_STANDARD 17)

# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Utils
add_library(utils SHARED utils/Utils.cpp)
target_link_libraries(utils Eigen3::Eigen)

set(LIBS Eigen3::Eigen utils)

# Directories
include_directories(utils)

# Executables
add_executable(parser parser.cpp)
target_link_libraries(parser ${LIBS})