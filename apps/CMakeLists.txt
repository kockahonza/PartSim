# All apps in the project will be using the library
link_libraries(PartSimLib)

find_package(HighFive)
find_package(libconfig)

# Build the gui apps (requiring SFML) if PARTSIM_GUI is not "nogui"
if(NOT (PARTSIM_GUI STREQUAL "nogui"))
    find_package(SFML REQUIRED COMPONENTS window graphics system)

    add_executable(TwoBodyGravity TwoBodyGravity.cpp)
    add_executable(LennardJones LennardJones.cpp)

    target_link_libraries(TwoBodyGravity sfml-graphics)
    target_link_libraries(LennardJones sfml-graphics config++)
endif()

add_executable(LennardJonesFast LennardJonesFast.cpp)
target_link_libraries(LennardJonesFast HighFive config++)

add_executable(LennardJonesFast3D LennardJonesFast3D.cpp)
target_link_libraries(LennardJonesFast3D HighFive config++)
