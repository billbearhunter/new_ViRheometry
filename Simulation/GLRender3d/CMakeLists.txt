cmake_minimum_required(VERSION 3.10)
project(GLRender3d)

# 包含源文件列表
include( CMakeSourceFiles.txt )

# 设置模块路径
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# C++ 标准设置
set(CMAKE_CXX_STANDARD 17)
add_definitions( -std=c++17 -Wall )

# 生成时间戳
string( TIMESTAMP BUILD_DATE_TIME "%Y-%m-%d %H:%M:%S" )
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/CompileDefinitions.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/CompileDefinitions.h" @ONLY )

# 定义可执行文件
add_executable( GLRender3d ${Headers} ${Sources} )

# 编译器检测
if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
  target_compile_definitions( GLRender3d PUBLIC CMAKE_DETECTED_GNU_COMPILER )
elseif( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" )
  target_compile_definitions( GLRender3d PUBLIC CMAKE_DETECTED_INTEL_COMPILER )
elseif( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
  target_compile_definitions( GLRender3d PUBLIC CMAKE_DETECTED_CLANG_COMPILER )
elseif( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" )
  target_compile_definitions( GLRender3d PUBLIC CMAKE_DETECTED_CLANG_COMPILER )
else()
  message( FATAL_ERROR "Unsupported C++ compiler ${CMAKE_CXX_COMPILER_ID}. Please email the maintainer." )
endif()

# --------------------------------------------------------
# 修复 Eigen3 链接
# --------------------------------------------------------
find_package( Eigen3 REQUIRED )

if(TARGET Eigen3::Eigen)
    # 现代 CMake 方式：直接链接目标
    target_link_libraries(GLRender3d Eigen3::Eigen)
else()
    # 备用方式：手动指定路径
    message(STATUS "Eigen3 target not found, using include dirs: ${EIGEN3_INCLUDE_DIRS}")
    target_include_directories(GLRender3d SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIRS})
    # 针对 Apple Silicon Homebrew 的强制兜底路径
    target_include_directories(GLRender3d SYSTEM PUBLIC "/opt/homebrew/include/eigen3")
endif()

# OpenCV 依赖
find_package( OpenCV REQUIRED )
target_include_directories( GLRender3d SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( GLRender3d ${OpenCV_LIBS} )

# OpenGL 和 GLUT 依赖
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
target_link_libraries(GLRender3d ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} )