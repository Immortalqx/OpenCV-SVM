cmake_minimum_required(VERSION 3.17)
project(SVMTrainer)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

add_executable(SVMTrainer main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})