cmake_minimum_required(VERSION 3.20)

project(hip_helper LANGUAGES HIP)

add_library(hip_helper INTERFACE
	hip_helper.hpp
)

target_link_libraries(hip_helper INTERFACE cpp_helper)

find_package(hip REQUIRED)

target_link_libraries(hip_helper INTERFACE hip::host hip::device)
