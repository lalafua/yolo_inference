cmake_minimum_required(VERSION 3.15)
project(yolo_prj)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_path(ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime)
find_library(ORT_LIB NAMES onnxruntime)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ORT_INCLUDE_DIR})

add_executable(${PROJECT_NAME} main.cpp src/detector.cpp src/gui.cpp include/gui.hpp include/detector.hpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ORT_LIB} Qt6::Core Qt6::Gui Qt6::Widgets)
