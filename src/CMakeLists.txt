add_library(PartSimLib SHARED PartSim.cpp util.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    find_package(OpenMP REQUIRED)
    target_link_libraries(PartSimLib PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(PartSimLib PUBLIC ../include)
