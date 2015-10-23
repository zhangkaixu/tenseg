cmake_minimum_required (VERSION 2.6)
project (Tenseg)

# check for c++11 suport
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#find_package(gflags REQUIRED)


#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set (SOURCE_DIR             ${PROJECT_SOURCE_DIR}/src)

include_directories("${SOURCE_DIR}")

# add the executable
add_executable(char_segger ${SOURCE_DIR}/char_segger/char_segger.cc)

add_executable(segtag ${SOURCE_DIR}/segtag.cc)
target_link_libraries(segtag gflags)
target_link_libraries(segtag glog)
