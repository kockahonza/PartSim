add_library(PartSimLib SHARED PartSim.cpp util.cpp)

target_include_directories(PartSimLib PUBLIC ../include)

find_package(Eigen3 REQUIRED)
target_link_libraries(PartSimLib PUBLIC Eigen3::Eigen)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    find_package(OpenMP REQUIRED)
    target_link_libraries(PartSimLib PRIVATE OpenMP::OpenMP_CXX)
endif()
