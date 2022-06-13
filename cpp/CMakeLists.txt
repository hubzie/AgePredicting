cmake_minimum_required(VERSION 3.22)
project(cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

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
add_library(utils SHARED utils/Utils.cpp)
target_link_libraries(utils Eigen3::Eigen)

# PCA
add_library(lpca SHARED utils/PCA.cpp)
target_link_libraries(lpca Eigen3::Eigen utils)

add_library(lkernelpca SHARED utils/KernelPCA.cpp)
target_link_libraries(lkernelpca Eigen3::Eigen utils)

# Models
add_library(model SHARED model/Model.cpp)
target_link_libraries(model Eigen3::Eigen)

# SVM
add_library(lsvm SHARED svm/LinearSVM.cpp
        svm/MultiClassLinearSVM.cpp
        svm/KernelSVM.cpp
        svm/MultiClassKernelSVM.cpp
        )
target_link_libraries(lsvm model utils)

# Ridge
add_library(lridge SHARED ridge/RidgeRegression.cpp ridge/KernelRidgeRegression.cpp)
target_link_libraries(lridge utils)

# CNN
add_library(lcnn SHARED cnn/CNN.cpp cnn/RegressionCNN.cpp
        cnn/layers/AveragePoolingLayer.cpp
        cnn/layers/ConvolutionalLayer.cpp
        cnn/layers/FullConnectedLayer.cpp
        cnn/layers/MaxPoolingLayer.cpp
        cnn/layers/ReLuLayer.cpp
        cnn/layers/ReshapeLayer.cpp
        cnn/layers/SigmoidLayer.cpp
        )
target_link_libraries(lcnn Eigen3::Eigen)

# Directories
include_directories(utils model svm cnn ridge)

# Libraries
set(LIBS Eigen3::Eigen utils model lpca lkernelpca lsvm lcnn lridge)

# Executables
add_executable(parser parser.cpp)
target_link_libraries(parser ${LIBS})

add_executable(svm_demo svm_demo.cpp)
target_link_libraries(svm_demo ${LIBS})

add_executable(kernel_pca_demo kernel_pca_demo.cpp)
target_link_libraries(kernel_pca_demo ${LIBS})

add_executable(pca_demo pca_demo.cpp)
target_link_libraries(pca_demo ${LIBS})

add_executable(cnn_demo cnn_demo.cpp)
target_link_libraries(cnn_demo ${LIBS})

add_executable(reg_cnn_demo regression_cnn_demo.cpp)
target_link_libraries(reg_cnn_demo ${LIBS})

# Generating data
add_executable(prepare_data prepare_data.cpp)
target_link_libraries(prepare_data ${LIBS})

add_executable(run_lsvm run_lsvm.cpp)
target_link_libraries(run_lsvm ${LIBS})

add_executable(run_ksvm run_ksvm.cpp)
target_link_libraries(run_ksvm ${LIBS})

add_executable(run_nn run_nn.cpp)
target_link_libraries(run_nn ${LIBS})

add_executable(run_regression_nn run_regression_nn.cpp)
target_link_libraries(run_regression_nn ${LIBS})

add_executable(run_ridge run_ridge.cpp)
target_link_libraries(run_ridge ${LIBS})
