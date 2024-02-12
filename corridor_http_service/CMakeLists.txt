cmake_minimum_required(VERSION 3.9)
project(HTTP_SERVICE)

# set variables
set(CMAKE_BUILD_TYPE "Release")
set(CGAL_DIR "$ENV{HOME}/CGAL-5.5.2")

find_package(CGAL REQUIRED)
find_package(cpprestsdk REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(Boost 1.45.0 COMPONENTS filesystem)
find_package(assimp REQUIRED)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

add_executable(server2 server2.cpp)

file(GLOB GEOMETRY_SRC_FILES geometry/*.cpp)
add_library(geometry ${GEOMETRY_SRC_FILES})
target_include_directories(geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/geometry)
target_link_libraries(geometry PUBLIC Eigen3::Eigen)

#new added
target_link_libraries(geometry PUBLIC CGAL::CGAL)
# target_link_libraries(geometry PUBLIC ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})

target_link_libraries(server2 Eigen3::Eigen)
target_link_libraries(server2 CGAL::CGAL)
target_link_libraries(server2 geometry)
target_link_libraries(server2 cpprestsdk::cpprest)
target_link_libraries(server2 assimp)



# add_executable(server server.cpp)
# target_link_libraries(server PRIVATE cpprestsdk::cpprest)
# target_link_libraries (server Eigen3::Eigen)
# target_link_libraries(server PRIVATE geometry)
## target_link_libraries(server ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})




if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    target_link_libraries(server2 ${Boost_LIBRARIES})
endif()


#add_executable(client client.cpp)
#target_link_libraries(client PRIVATE cpprestsdk::cpprest)

