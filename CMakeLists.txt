cmake_minimum_required(VERSION 3.15)
include(FetchContent)
set (PROJECT VisualOdometry) 
project (${PROJECT})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV 4.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

FetchContent_Declare(
        readerwriterqueue
        GIT_REPOSITORY    https://github.com/cameron314/readerwriterqueue
        GIT_TAG           master
)
FetchContent_MakeAvailable(readerwriterqueue)

find_package(Threads REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(./include)

file(GLOB MyCSources ./src/*.cpp)
set(SOURCES ${MyCSources})
source_group("Source Files" FILES ${SOURCES})

add_executable( ${PROJECT}  ${HEADERS} ${SOURCES} )
target_link_libraries( ${PROJECT} ${OpenCV_LIBS} Eigen3::Eigen readerwriterqueue pango_display)
set_property(TARGET ${PROJECT} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
