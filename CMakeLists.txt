cmake_minimum_required(VERSION 3.10)
set (PROJECT VisualOdometry) 
project (${PROJECT})

find_package( OpenCV 4.0 REQUIRED )
#Eigen3
find_package(Eigen3 3.3.0 REQUIRED NO_MODULE)
if (Eigen3_FOUND)
  message(STATUS "Found Eigen3 lib : ${Eigen3_VERSION} - ${EIGEN3_INCLUDE_DIRS}")
endif (Eigen3_FOUND)

#OpenGV
find_package(opengv REQUIRED)
# Bug include CMake 1.17
# get_target_property(opengv_INCLUDE_DIR opengv INTERFACE_INCLUDE_DIRECTORIES)

if (opengv_FOUND)
  message(STATUS "Found Opengv lib : ${opengv_INCLUDE_DIR} - ${opengv_DIR}")
  add_definitions(-DUSE_OPENGV)
endif (opengv_FOUND)

#Sophus
# First search for a custom build of Sophus within this project
find_package(Sophus NO_DEFAULT_PATH PATHS ${PROJECT_SOURCE_DIR}/external/Sophus/build/)
find_package(Sophus REQUIRED)

if (Sophus_FOUND)
  message(STATUS "Found Sophus lib : ${Sophus_VERSION} - ${Sophus_DIR}")
endif (Sophus_FOUND)

include_directories( ${OpenCV_INCLUDE_DIRS} )

FILE(GLOB MyHeaders ./src/*.h)
FILE(GLOB MyCSources ./src/*.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (HEADERS
    ${MyHeaders}
    )

set (SOURCES 
    ${MyCSources}
    )
        
source_group ("Header Files" FILES ${HEADERS})
source_group ("Source Files" FILES ${SOURCES})

add_executable( ${PROJECT}  ${HEADERS} ${SOURCES} )
target_link_libraries( ${PROJECT} ${OpenCV_LIBS} )
