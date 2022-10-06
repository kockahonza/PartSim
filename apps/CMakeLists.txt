# All apps in the project will be using the library
link_libraries(PartSimLib)

# Build the gui apps (requiring SFML) if PARTSIM_GUI is not "nogui"
if(NOT (PARTSIM_GUI STREQUAL "nogui"))
    find_package(SFML REQUIRED COMPONENTS window graphics system)

    add_executable(TwoBodyGravity TwoBodyGravity.cpp)
    add_executable(LennardJones LennardJones.cpp)

    target_link_libraries(TwoBodyGravity sfml-graphics)
    target_link_libraries(LennardJones sfml-graphics)
endif()

find_package(HighFive)
find_package(libconfig)

add_executable(LennardJonesFast LennardJonesFast.cpp)
target_link_libraries(LennardJonesFast HighFive config)
