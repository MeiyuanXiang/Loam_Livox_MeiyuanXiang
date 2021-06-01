#include <ros/ros.h>
#include <fstream>
#include <sensor_msgs/Image.h>
#include <string>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

std::string img_path;
std::vector<std::string> img_list_vec;
int selected_frame_idx = -1;
double first_received_time;
bool fisrt_align = true;

ros::Publisher pub_color_map;

double str2double(std::string str)
{
    stringstream ss;
    ss << str;
    double timestamp;
    ss >> timestamp;

    return timestamp;
}

string time2str(double timestamp)
{
    stringstream ss;
    ss << fixed << timestamp;
    string str;
    ss >> str;
    
    return str;
}

void CallbackHandler(const sensor_msgs::Image::ConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat img = cv_ptr->image;

    double timestamp = msg->header.stamp.toSec();
    string timestamp_str = time2str(timestamp);
    ROS_INFO("Timestamp: %s\n", timestamp_str.c_str());

    cv::imwrite("/home/zjucvg/Desktop/Data/8-3/sync/part4/img/" + timestamp_str + ".png", img);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "extract_image");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe<sensor_msgs::Image>("/camera/color/image_raw", 10000, &CallbackHandler);

    ros::spin();

    return 0;
}