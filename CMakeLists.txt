cmake_minimum_required(VERSION 3.11)

project(
  PartSim
  VERSION 0.1
  DESCRIPTION "A very simple particle simulation library along with some demos"
  LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

endif()

set(HIGHFIVE_USE_EIGEN ON)
add_subdirectory(external_libs/HighFive)

add_subdirectory(src)

add_subdirectory(apps)
