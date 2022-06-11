cmake_minimum_required(VERSION 3.22)
project(cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Multithreading
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Utils
add_library(utils SHARED utils/Utils.cpp utils/PCA.cpp)
target_link_libraries(utils Eigen3::Eigen)

# Models
add_library(model SHARED model/Model.cpp)
target_link_libraries(model Eigen3::Eigen)

# SVM
add_library(lsvm SHARED svm/LinearSVM.cpp svm/MultiClassLinearSVM.cpp)
target_link_libraries(lsvm model utils)

# CNN
add_library(lcnn SHARED cnn/CNN.cpp
        cnn/layers/FullConnectedLayer.cpp
        cnn/layers/SigmoidLayer.cpp
        )
target_link_libraries(lcnn Eigen3::Eigen)

# Directories
include_directories(utils model svm cnn)

# Libraries
set(LIBS Eigen3::Eigen utils model lsvm lcnn)

# Executables
add_executable(parser parser.cpp)
target_link_libraries(parser ${LIBS})

add_executable(svm_demo svm_demo.cpp)
target_link_libraries(svm_demo ${LIBS})

add_executable(pca_demo pca_demo.cpp)
target_link_libraries(pca_demo ${LIBS})

add_executable(cnn_demo cnn_demo.cpp)
target_link_libraries(cnn_demo ${LIBS})