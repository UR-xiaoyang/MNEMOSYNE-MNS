cmake_minimum_required(VERSION 3.20)
project(mnemosyne_gateway LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(mnemosyne_gateway
    gateway.cpp
)

target_compile_definitions(mnemosyne_gateway PRIVATE
    MNEMOSYNE_GATEWAY_VERSION="1.0.0-native"
)

if(MSVC)
    target_compile_options(mnemosyne_gateway PRIVATE /W4 /EHsc)
else()
    target_compile_options(mnemosyne_gateway PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wno-cast-function-type)
endif()
target_link_libraries(mnemosyne_gateway PRIVATE ws2_32)

# SQLite is loaded dynamically at runtime. This keeps the binary buildable
# without a package manager while preserving a Python fallback deployment.
install(TARGETS mnemosyne_gateway RUNTIME DESTINATION bin)
