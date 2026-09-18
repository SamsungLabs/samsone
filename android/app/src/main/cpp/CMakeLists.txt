cmake_minimum_required(VERSION 3.22.1)
project("samsone")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_library(log-lib log)

# Inject the log library into the global linker flags.
# This ensures that even third-party static libs (like sentencepiece/protobuf)
# can find __android_log_write during the final link phase.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -llog")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -llog")

add_subdirectory(tokenizers-cpp)

add_library(native-lib SHARED native-lib.cpp)

target_link_libraries(native-lib
        PRIVATE
        tokenizers_cpp
        sentencepiece-static
        ${log-lib}
)