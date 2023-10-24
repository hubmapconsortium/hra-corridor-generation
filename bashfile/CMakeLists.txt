cmake_minimum_required(VERSION 3.9)
project(CORRIDOR)

# set variables
set(CMAKE_BUILD_TYPE "Release")
set(CGAL_DIR "$ENV{HOME}/CGAL-5.5.2")

 message("${CGAL_DIR}")

# find packages
find_package(CGAL REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(Boost 1.45.0 COMPONENTS filesystem)
find_package(cpprestsdk REQUIRED)
find_package(OpenMP REQUIRED)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 


add_executable(corridor_generator main.cpp)
#add_executable(corridor_api server2.cpp)
#add_executable(corridor_components number_of_components.cpp)

file(GLOB GEOMETRY_SRC_FILES geometry/*.cpp)
add_library(geometry ${GEOMETRY_SRC_FILES})
target_include_directories(geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/geometry)
target_link_libraries(geometry PUBLIC CGAL::CGAL)
target_link_libraries(geometry PUBLIC Eigen3::Eigen)

#1
target_link_libraries(geometry PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(corridor_generator Eigen3::Eigen)
target_link_libraries(corridor_generator CGAL::CGAL)
target_link_libraries(corridor_generator geometry)

#2
target_link_libraries(corridor_generator OpenMP::OpenMP_CXX)

#target_link_libraries(corridor_api Eigen3::Eigen)
#target_link_libraries(corridor_api CGAL::CGAL)
#target_link_libraries(corridor_api geometry)
#target_link_libraries(corridor_api cpprestsdk::cpprest)
#3
#target_link_libraries(corridor_api OpenMP::OpenMP_CXX)

#target_link_libraries(corridor_components Eigen3::Eigen)
#target_link_libraries(corridor_components CGAL::CGAL)

if (Boost_FOUND)
    target_link_libraries(corridor_generator ${Boost_LIBRARIES})
    #target_link_libraries(corridor_api ${Boost_LIBRARIES})
    #target_link_libraries(corridor_components ${Boost_LIBRARIES})
endif()