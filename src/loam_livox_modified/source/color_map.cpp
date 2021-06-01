#include "tools/common.h"
#include <ros/ros.h>
#include <fstream>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64.h>
#include <string>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <unistd.h>

using namespace std;

struct Pose
{
    Eigen::Quaterniond Qw;
    Eigen::Vector3d Tw;
};

std::vector<Pose> idx_to_odom_vec;

std::string img_path;
std::vector<std::string> img_list_vec;
std::vector<double> img_list_double_vec;
std::vector<std::string> odom_list_vec;
std::vector<double> odom_list_double_vec;
int selected_img_frame_idx = -1;
int selected_lidar_frame_idx = -1;
int last_frame_idx = -1;
double first_received_time = -1;

ros::Publisher pub_color_map;
cv::Mat unImg;

cv::Mat K = (cv::Mat_<float>(3, 3) << 903.2916, 0, 636.1603,
             0, 905.1799, 358.7213,
             0.0, 0.0, 1.0);
cv::Mat Tcl = (cv::Mat_<float>(3, 4) << 0.0096, -0.9999, 0.0131, 0.0681,
               0.0300, -0.0128, -0.9995, 0.0068,
               0.9995, 0.0100, 0.0298, -0.0523);

cv::Mat DistortCoeff = (cv::Mat_<float>(4, 1) << 0.1287, -0.2158, 0, 0);

ofstream log_file("/home/zjucvg/log7.txt");
ofstream log_file1("/home/zjucvg/log6.txt");
ofstream log_file2("/home/zjucvg/log9.txt");

double str2double(std::string str)
{
    std::stringstream ss;
    ss << fixed << str;
    double timestamp;
    ss >> timestamp;
    return timestamp;
}

std::string double2str(double timestamp)
{
    std::stringstream ss;
    ss << fixed << timestamp;
    std::string str;
    ss >> str;
    return str;
}

void log_out_pose(ostream &out, Pose pose, int idx)
{
    out << "pose" << idx << " :" << pose.Qw.x() << "  " << pose.Qw.y() << "  " << pose.Qw.z() << "  " << pose.Qw.w() << "  " << pose.Tw.x() << "  " << pose.Tw.y() << "  " << pose.Tw.z() << endl;
}

cv::Mat Quaterniond2Mat(Eigen::Quaterniond &q)
{
    cv::Mat res = cv::Mat(3, 3, CV_32FC1);

    res.at<float>(0, 0) = 1 - 2 * q.y() * q.y() - 2 * q.z() * q.z();
    res.at<float>(0, 1) = 2 * q.x() * q.y() + 2 * q.w() * q.z();
    res.at<float>(0, 2) = 2 * q.x() * q.z() - 2 * q.w() * q.y();
    res.at<float>(1, 0) = 2 * q.x() * q.y() - 2 * q.w() * q.z();
    res.at<float>(1, 1) = 1 - 2 * q.x() * q.x() - 2 * q.z() * q.z();
    res.at<float>(1, 2) = 2 * q.y() * q.z() + 2 * q.w() * q.x();
    res.at<float>(2, 0) = 2 * q.x() * q.z() + 2 * q.w() * q.y();
    res.at<float>(2, 1) = 2 * q.y() * q.z() - 2 * q.w() * q.x();
    res.at<float>(2, 2) = 1 - 2 * q.x() * q.x() - 2 * q.y() * q.y();

    return res.clone();
}

bool isFirst = true;

void CallbackHandler(const sensor_msgs::PointCloud2ConstPtr &msgs)
{
    pcl::PointCloud<pcl::PointXYZRGB> laserPointCloudColor;
    sensor_msgs::PointCloud2 PointCloudColorOut;

    pcl::PointCloud<pcl::PointXYZI> PointCloudIn;
    pcl::fromROSMsg(*msgs, PointCloudIn);

    int pointcloud_size = PointCloudIn.points.size();

    double timestamp1 = PointCloudIn.points[0].intensity;

    bool first = true;
    for (int i = 0; i < pointcloud_size; i++)
    {
        double timestamp = PointCloudIn.points[i].intensity + first_received_time;
        for (int j = 0; j < img_list_double_vec.size() - 1; j++)
        {
            if (timestamp < img_list_double_vec[0])
            {
                selected_img_frame_idx = 0;
                break;
            }
            else if (timestamp > img_list_double_vec[j] && timestamp < img_list_double_vec[j + 1])
            {
                selected_img_frame_idx = timestamp > (img_list_double_vec[j] / 2.0 + img_list_double_vec[j + 1] / 2.0) ? j + 1 : j;
                break;
            }
            else if (timestamp > img_list_double_vec[img_list_double_vec.size() - 1])
            {
                selected_img_frame_idx = img_list_double_vec.size() - 1;
                break;
            }
        }

        log_file << "img frame idx: " << selected_img_frame_idx << "   " << img_list_vec[selected_img_frame_idx] << endl;

        if (last_frame_idx != selected_img_frame_idx)
        {
            last_frame_idx = selected_img_frame_idx;
            cv::Mat img = cv::imread(img_path + "img/" + img_list_vec[selected_img_frame_idx] + ".png");
            cv::undistort(img, unImg, K, DistortCoeff);
        }

        double frame_timestamp = img_list_double_vec[selected_img_frame_idx];
        std::string frame_timestamp_str = img_list_vec[selected_img_frame_idx];

        selected_lidar_frame_idx = -1;

        for (int j = 0; j < odom_list_double_vec.size() - 1; j++)
        {
            if (frame_timestamp >= odom_list_double_vec[j] && frame_timestamp <= odom_list_double_vec[j + 1])
            {
                selected_lidar_frame_idx = j;
            }
        }

        if (selected_lidar_frame_idx == -1)
        {
            log_file << "lidar idx: " << selected_lidar_frame_idx << "   " << odom_list_vec[selected_lidar_frame_idx] << endl;
            continue;
        }

        Pose &pose1 = idx_to_odom_vec[selected_lidar_frame_idx];
        Pose &pose2 = idx_to_odom_vec[selected_lidar_frame_idx + 1];

        double t = (frame_timestamp - odom_list_double_vec[selected_lidar_frame_idx]) / (odom_list_double_vec[selected_lidar_frame_idx + 1] - odom_list_double_vec[selected_lidar_frame_idx]);

        Pose cam_pose;
        cam_pose.Qw = pose1.Qw.slerp(t, pose2.Qw);
        cam_pose.Tw = pose1.Tw + t * (pose2.Tw - pose1.Tw);

        cv::Mat Rwl = Quaterniond2Mat(cam_pose.Qw);
        cv::Mat twl = (cv::Mat_<float>(3, 1) << cam_pose.Tw.x(), cam_pose.Tw.y(), cam_pose.Tw.z());
        cv::Mat Rlw, tlw;

        Rlw = Rwl;
        tlw = -Rlw * twl;

        cv::Mat Tlw = cv::Mat::eye(4, 4, CV_32FC1);

        Rlw.copyTo(Tlw.colRange(0, 3).rowRange(0, 3));
        Tlw.at<float>(0, 3) = tlw.at<float>(0, 0);
        Tlw.at<float>(1, 3) = tlw.at<float>(1, 0);
        Tlw.at<float>(2, 3) = tlw.at<float>(2, 0);

        cv::Mat Tcw = Tcl * Tlw;

        pcl::PointXYZI &pt = PointCloudIn.points[i];
        cv::Mat pos = (cv::Mat_<float>(4, 1) << pt.x, pt.y, pt.z, 1.0);

        cv::Mat pixel = K * Tcw * pos;

        int x = cvRound(pixel.at<float>(0, 0) / pixel.at<float>(2, 0));
        int y = cvRound(pixel.at<float>(1, 0) / pixel.at<float>(2, 0));

        int rgb[3] = {255, 255, 255};
        if (x > 0 && x < unImg.cols && y > 0 && y < unImg.rows)
        {
            rgb[0] = (int)unImg.at<cv::Vec3b>(y, x)[2];
            rgb[1] = (int)unImg.at<cv::Vec3b>(y, x)[1];
            rgb[2] = (int)unImg.at<cv::Vec3b>(y, x)[0];
        }

        log_file << rgb[0] << " " << rgb[1] << " " << rgb[2] << endl;

        pcl::PointXYZRGB ptc;

        ptc.x = pt.x;
        ptc.y = pt.y;
        ptc.z = pt.z;
        ptc.r = rgb[0];
        ptc.g = rgb[1];
        ptc.b = rgb[2];

        laserPointCloudColor.push_back(ptc);
    }

    cv::imwrite("/home/zjucvg/outcolor.png", unImg);
    pcl::toROSMsg(laserPointCloudColor, PointCloudColorOut);
    PointCloudColorOut.header.stamp = msgs->header.stamp;
    PointCloudColorOut.header.frame_id = "camera_init";
    pub_color_map.publish(PointCloudColorOut);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "map_with_color");
    ros::NodeHandle nh;

    nh.param<std::string>("common/img_path", img_path, "/home/zjucvg/Desktop/Data/");

    ifstream img_list(img_path + "img_list.txt");
    std::string line;
    while (getline(img_list, line))
    {
        img_list_vec.push_back(line);
        img_list_double_vec.push_back(str2double(line));
    }

    ifstream odom_list(img_path + "odom_res.txt");
    getline(odom_list, line);
    first_received_time = str2double(line);
    while (getline(odom_list, line))
    {
        string time;
        double qx, qy, qz, qw, tx, ty, tz;
        Pose pose;
        stringstream ss;
        ss << line;
        ss >> time >> qx >> qy >> qz >> qw >> tx >> ty >> tz;
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Vector3d t(tx, ty, tz);
        pose.Qw = q;
        pose.Tw = t;
        idx_to_odom_vec.push_back(pose);
        odom_list_double_vec.push_back(str2double(time));
        odom_list_vec.push_back(time);
    }

    ros::Subscriber sub_pc = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 10000, &CallbackHandler);

    pub_color_map = nh.advertise<sensor_msgs::PointCloud2>("/map_with_color", 10000);

    ros::spin();

    return 0;
}