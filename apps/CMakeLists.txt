find_package(SFML REQUIRED COMPONENTS window graphics system)

# All apps in the project will be using the library
link_libraries(PartSimLib)

add_executable(TwoBodyGravity TwoBodyGravity.cpp)
add_executable(LennardJones LennardJones.cpp)

target_link_libraries(TwoBodyGravity sfml-graphics)
target_link_libraries(LennardJones sfml-graphics)
