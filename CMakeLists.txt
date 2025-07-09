cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_CXX_STANDARD 20)
project(DL_TensorRT_test)

# 包含头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
# 添加库文件目录
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 查找OpenCV库
if (WIN32)
    # 设置 OpenCV 路径
    set(OpenCV_DIR "D:/Programs/opencv/build")
endif ()
find_package(OpenCV REQUIRED)


# 需要将所有使用到的 cpp 文件添加进去
add_executable(deep_learning
        src/tools.cpp
        test/test_main.cpp
        test/test_segment.cpp
)
# 链接OpenCV库
target_link_libraries(deep_learning
        dl_base
        ${OpenCV_LIBS}
)

if (WIN32)
    # Windows 需要手动复制 DLL 到输出目录
    add_custom_command(TARGET deep_learning POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/lib/dl_base.dll"
            $<TARGET_FILE_DIR:deep_learning>
            COMMENT "Copying DLL to output directory..."
    )
endif ()