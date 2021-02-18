cmake_minimum_required(VERSION 3.10)
set (PROJECT VisualOdometry) 
project (${PROJECT})

find_package( OpenCV 4.0 REQUIRED )

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
