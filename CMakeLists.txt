cmake_minimum_required (VERSION 3.6)
project (ProgParallel)
find_package(Threads REQUIRED)
find_package(SDL2 REQUIRED)
find_package(benchmark REQUIRED)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-W -Wall -mavx -mavx2 -msse4.2")
set(CMAKE_CXX_RELEASE_FLAGS "${CMAKE_CXX_RELEASE_FLAGS} -O3 -march=native -mavx -mavx2 -msse4.2")

include_directories(lib/include)

add_library(renderer src/render.hpp src/render.cpp)
add_executable(view src/view.cpp)

target_link_libraries(view renderer SDL2::SDL2 Threads::Threads tbb)

link_directories(/usr/lib)

add_executable(bench src/bench.cpp)
target_link_libraries(bench renderer benchmark::benchmark tbb)
