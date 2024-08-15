cmake_minimum_required(VERSION 3.9)
project(HTTP_SERVICE)

# set variables
set(CMAKE_BUILD_TYPE "Release")
set(CGAL_DIR "$ENV{HOME}/CGAL-5.5.3")


set(CMAKE_C_COMPILER /usr/bin/gcc-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
find_package(CUDA QUIET REQUIRED)

find_package(CGAL REQUIRED)
find_package(cpprestsdk REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(Boost 1.45.0 COMPONENTS filesystem)
find_package(assimp REQUIRED)
find_package(OpenMP REQUIRED)


set(CMAKE_CXX_STANDARD 14)
# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

cuda_select_nvcc_arch_flags(ARCH_FLAGS)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 


file(GLOB GEOMETRY_SRC_FILES geometry/*.cpp)
add_library(geometry ${GEOMETRY_SRC_FILES})
target_include_directories(geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/geometry)
target_link_libraries(geometry PUBLIC Eigen3::Eigen)
target_link_libraries(geometry PUBLIC CGAL::CGAL)
target_link_libraries(geometry PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(geometry PUBLIC assimp)

# static library corridor_gpu
set(CUDA_SEPARABLE_COMPILATION ON)
file(GLOB CORRIDOR_GPU_SRC_FILES corridor_gpu/*.cu)
cuda_add_library(corridor_gpu ${CORRIDOR_GPU_SRC_FILES})
set_property(TARGET corridor_gpu PROPERTY CUDA_ARCHITECTURES native)
target_include_directories(corridor_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/corridor_gpu)
target_link_libraries(corridor_gpu ${CUDA_LIBRARIES})

#static library corridor_cpu
file(GLOB CORRIDOR_CPU_SRC_FILES corridor_cpu/*.cpp)
add_library(corridor_cpu ${CORRIDOR_CPU_SRC_FILES})
target_include_directories(corridor_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/corridor_cpu)
target_link_libraries(corridor_cpu geometry)

# target_link_libraries(geometry PUBLIC ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})


add_executable(corridor_api server2.cpp)
target_link_libraries(corridor_api CGAL::CGAL)
target_link_libraries(corridor_api geometry)
target_link_libraries(corridor_api cpprestsdk::cpprest)
target_link_libraries(corridor_api OpenMP::OpenMP_CXX)
target_link_libraries(corridor_api corridor_gpu)
target_link_libraries(corridor_api corridor_cpu)


# add_executable(server server.cpp)
# target_link_libraries(server PRIVATE cpprestsdk::cpprest)
# target_link_libraries (server Eigen3::Eigen)
# target_link_libraries(server PRIVATE geometry)
## target_link_libraries(server ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})




if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    target_link_libraries(corridor_api ${Boost_LIBRARIES})
endif()


