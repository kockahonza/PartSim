find_package(SFML REQUIRED COMPONENTS window graphics system)

add_executable(TwoBodyGravity TwoBodyGravity.cpp)
add_executable(LennardJones LennardJones.cpp)

target_link_libraries(TwoBodyGravity PartSimLib sfml-graphics)
target_link_libraries(LennardJones PartSimLib sfml-graphics)
