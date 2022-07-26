# 指定cmake的最低版本
cmake_minimum_required(VERSION 3.2)

# 指生成一个名为crop_row的工程
project(main)

# 添加OPENCV库
# 如果不需要指定OpenCV版本，如下
# find_package(OpenCV REQUIRED)会在系统中找到OpenCVConfig.cmake，该文件定义了OpenCV_INCLUDE_DIRS和OpenCV_LIBS 等变量。
find_package(OpenCV REQUIRED)
# 如果需要指定OpenCV版本，如下
# find_package(OpenCV 3.3 REQUIRED)

# 添加OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/home/xnwu/wxn/darknet/src)

# 显示OpenCV_INCLUDE_DIRS的值
message(${OpenCV_INCLUDE_DIRS})

# 添加一个可执行程序，基于`pengzhangfushi.cpp`生成名为`pengzhang`的可执行文件。
# 语法：add_executable( 程序名 源代码文件 ）
add_executable(main main.cpp improcess.cpp)

# 将库文件（此处是opencv的库文件）链接到可执行程序上。
target_link_libraries(main /home/xnwu/wxn/darknet/libdarknet.so) # 显示指定链接动态库
target_link_libraries(main  ${OpenCV_LIBS})
