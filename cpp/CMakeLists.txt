cmake_minimum_required(VERSION 3.22)
project(cpp)
set(CMAKE_CXX_STANDARD 17)

# Multithreading
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Utils
add_library(utils SHARED utils/Utils.cpp utils/PCA.cpp)
target_link_libraries(utils Eigen3::Eigen)

# Models
add_library(model SHARED model/Model.cpp)
target_link_libraries(model Eigen3::Eigen)

# SVM
add_library(lsvm SHARED svm/LinearSVM.cpp)
target_link_libraries(lsvm model)

# Libraries
set(LIBS Eigen3::Eigen utils model lsvm)

# Directories
include_directories(utils model svm)

# Executables
add_executable(parser parser.cpp)
target_link_libraries(parser ${LIBS})

add_executable(svm svm.cpp)
target_link_libraries(svm ${LIBS})

add_executable(pca_demo pca_demo.cpp)
target_link_libraries(pca_demo ${LIBS})