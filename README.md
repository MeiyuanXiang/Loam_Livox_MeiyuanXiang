# Loam_Livox_MeiyuanXiang
Loam_Livox相关论文、代码中文注释以及代码改动

# 环境
1. Ubuntu（测试了Ubuntu16.04.5、Ubuntu18.04）
2. ROS (测试了kinetic、melodic)
3. PCL（测试了pcl1.9）
4. Opencv（测试了opencv3.4.3）
5. Ceres Solver（测试ceres-solver-1.14.0）
6. livox_ros_driver

# 编译
1. 下载源码 git clone https://github.com/MeiyuanXiang/Loam_Livox_MeiyuanXiang.git
2. 将Loam_Livox_MeiyuanXiang\src下的loam_livox或loam_livox_modified拷贝到ros工程空间src文件夹内，例如~/catkin_ws/src/
3. cd ~/catkin_ws
4. catkin_make
5. source ~/catkin_ws/devel/setup.bash

# 数据
链接：https://pan.baidu.com/s/1nMSJRuP8io8mEqLgACUT_w
提取码：sv9z

# 运行
1. Livox Mid-40、Livox Mid-70直连
roslaunch loam_livox livox.launch
roslaunch livox_ros_driver livox_lidar.launch
2. 普通场景数据
roslaunch loam_livox rosbag.launch
rosbag play YOUR_DOWNLOADED.bag
3. 大场景数据
roslaunch loam_livox rosbag_largescale.launch
rosbag play YOUR_DOWNLOADED.bag
4. Livox Mid-100数据
roslaunch loam_livox rosbag_mid100.launch
rosbag play mid100_example.bag
5. 简单回环检测
roslaunch loam_livox rosbag_loop_simple.launch
rosbag play YOUR_PATH/loop_simple.bag
6. 回环检测
roslaunch loam_livox rosbag_loop.launch
rosbag play YOUR_DOWNLOADED.bag
