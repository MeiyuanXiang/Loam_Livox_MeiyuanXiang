// This is the Lidar Odometry And Mapping (LOAM) for solid-state lidar (for example: livox lidar),
// which suffer form motion blur due the continously scan pattern and low range of fov.

// Developer: Jiarong Lin  ziv.lin.ljr@gmail.com

//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef LIVOX_LASER_SCAN_HANDLER_HPP
#define LIVOX_LASER_SCAN_HANDLER_HPP

#include <cmath>
#include <vector>

#define USE_HASH 1
#define SHOW_OPENCV_VIS 0
#if USE_HASH
#include <unordered_map>
#endif

#include <Eigen/Eigen>
#include <Eigen/Eigen>
#include <nav_msgs/Odometry.h>
#include <fstream>
#include <sstream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "tools/common.h"
#include "tools/pcl_tools.hpp"
#include "tools/tools_eigen_math.hpp"
#include "tools/tools_logger.hpp"
#define PCL_DATA_SAVE_DIR "/home/ziv/data/loam_pc"

#define IF_LIVOX_HANDLER_REMOVE 0
#define IF_APPEND 0
#define printf_line printf(" %s %d \r\n", __FILE__, __LINE__);

using namespace std;
using namespace PCL_TOOLS;

ofstream log_file1("/home/zjucvg/log2.txt");
ofstream log_file2("/home/zjucvg/log8.txt");
cv::Mat K = (cv::Mat_<float>(3, 3) << 903.2916, 0, 636.1603,
             0, 905.1799, 358.7213,
             0.0, 0.0, 1.0);
cv::Mat Tcl = (cv::Mat_<float>(3, 4) << 0.0096, -0.9999, 0.0131, 0.0681,
               0.0300, -0.0128, -0.9995, 0.0068,
               0.9995, 0.0100, 0.0298, -0.0523);

cv::Mat DistortCoeff = (cv::Mat_<float>(4, 1) << 0.1287, -0.2158, 0, 0);

class Livox_laser
{
public:
    string SOFT_WARE_VERSION = string("V_0.1_beta");

    enum E_point_type
    {
        e_pt_normal = 0,                      // normal points
        e_pt_000 = 0x0001 << 0,               // points [0,0,0]
        e_pt_too_near = 0x0001 << 1,          // points in short range
        e_pt_reflectivity_low = 0x0001 << 2,  // low reflectivity
        e_pt_reflectivity_high = 0x0001 << 3, // high reflectivity
        e_pt_circle_edge = 0x0001 << 4,       // points near the edge of circle
        e_pt_nan = 0x0001 << 5,               // points with infinite value
        e_pt_small_view_angle = 0x0001 << 6,  // points with large viewed angle
    };

    enum E_feature_type // if and only if normal point can be labeled
    {
        e_label_invalid = -1,
        e_label_unlabeled = 0,
        e_label_corner = 0x0001 << 0,
        e_label_surface = 0x0001 << 1,
        e_label_near_nan = 0x0001 << 2,
        e_label_near_zero = 0x0001 << 3,
        e_label_hight_intensity = 0x0001 << 4,
    };

    // Encode point infos using points intensity, which is more convenient for debugging.
    enum E_intensity_type
    {
        e_I_raw = 0,
        e_I_motion_blur,
        e_I_motion_mix,
        e_I_sigma,
        e_I_scan_angle,
        e_I_curvature,
        e_I_view_angle,
        e_I_time_stamp
    };

    struct Pt_infos
    {
        int pt_type = e_pt_normal;
        int pt_label = e_label_unlabeled;
        int idx = 0.f;
        float raw_intensity = 0.f;
        float time_stamp = 0.0;
        float polar_angle = 0.f;
        int polar_direction = 0;
        float polar_dis_sq2 = 0.f;
        float depth_sq2 = 0.f;
        float curvature = 0.0;
        float view_angle = 0.0;
        float sigma = 0.0;
        Eigen::Matrix<float, 2, 1> pt_2d_img; // project to X==1 plane
    };

    // E_intensity_type   default_return_intensity_type = e_I_raw;
    E_intensity_type default_return_intensity_type = e_I_motion_blur;
    // E_intensity_type default_return_intensity_type = e_I_scan_angle;
    // E_intensity_type default_return_intensity_type = e_I_curvature;
    // E_intensity_type default_return_intensity_type = e_I_view_angle;

    int pcl_data_save_index = 0;

    float max_fov = 17; // Edge of circle to main axis
    float m_max_edge_polar_pos = 0;
    float m_time_internal_pts = 1.0e-5; // 10us = 1e-5
    float m_cx = 0;
    float m_cy = 0;
    int m_if_save_pcd_file = 0; // 是否保存从ROS中得到的原始点云数据
    int m_input_points_size;
    double m_first_receive_time = -1;
    double m_current_time;
    double m_last_maximum_time_stamp;
    float thr_corner_curvature = 0.05;
    float thr_surface_curvature = 0.01;
    float minimum_view_angle = 10;
    std::vector<Pt_infos> m_pts_info_vec;
    std::vector<PointType> m_raw_pts_vec;

    std::vector<string> m_img_timestamp_vec;

    bool m_if_color_map;
    std::string m_img_path;

#if USE_HASH
    std::unordered_map<PointType, Pt_infos *, Pt_hasher, Pt_compare> m_map_pt_idx; // using hash_map
    std::unordered_map<PointType, Pt_infos *, Pt_hasher, Pt_compare>::iterator m_map_pt_idx_it;
#else
    std::map<PointType, Pt_infos *, Pt_compare> m_map_pt_idx;
    std::map<PointType, Pt_infos *, Pt_compare>::iterator m_map_pt_idx_it;
#endif

    float m_livox_min_allow_dis = 1.0;
    float m_livox_min_sigma = 7e-3;

    std::vector<pcl::PointCloud<pcl::PointXYZI>> m_last_laser_scan;

    int m_img_width = 800;
    int m_img_heigh = 800;
    ADD_SCREEN_PRINTF_OUT_METHOD;

    ~Livox_laser() {}

    Livox_laser()
    {
        // Some data init
        ENABLE_SCREEN_PRINTF;
        screen_out << "========= Hello, this is livox laser ========" << endl;
        screen_out << "Compile time:  " << __TIME__ << endl;
        screen_out << "Softward version: " << SOFT_WARE_VERSION << endl;
        screen_out << "========= End ========" << endl;

        log_file1 << "init livox" << endl;

        m_max_edge_polar_pos = std::pow(tan(max_fov / 57.3) * 1, 2);
    }

    template <typename T>
    T dis2_xy(T x, T y)
    {
        return x * x + y * y;
    }

    template <typename T>
    T depth2_xyz(T x, T y, T z)
    {
        return x * x + y * y + z * z;
    }

    template <typename T>
    T depth_xyz(T x, T y, T z)
    {
        return sqrt(depth2_xyz(x, y, z));
    }

    template <typename T>
    Pt_infos *find_pt_info(const T &pt)
    {
        m_map_pt_idx_it = m_map_pt_idx.find(pt); // std::map with custom compare, x->y->z order
        if (m_map_pt_idx_it == m_map_pt_idx.end())
        {
            printf("Input pt is [%lf, %lf, %lf]\r\n", pt.x, pt.y, pt.z);
            printf("Error!!!!\r\n");
            assert(m_map_pt_idx_it != m_map_pt_idx.end()); // else, there must be something error happened before.
        }

        return m_map_pt_idx_it->second;
    }

    double str2double(string str)
    {
        stringstream ss;
        ss << str;
        double res;
        ss >> res;

        return res;
    }

    void get_img_list()
    {
        ifstream img_timestamp_list(m_img_path + "img_list.txt");
        log_file1 << "path: " << m_img_path + "img_list.txt" << endl;
        std::string line;
        while (getline(img_timestamp_list, line))
        {
            m_img_timestamp_vec.push_back(line);
        }
    }

    void get_features(pcl::PointCloud<PointType> &pc_corners, pcl::PointCloud<PointType> &pc_surface, pcl::PointCloud<PointType> &pc_full_res,
                      pcl::PointCloud<ColorType> &pc_full_color,
                      float minimum_blur = 0.0,
                      float maximum_blur = 0.3)
    {
        cv::Mat unImg;
        int idx_select = -1;
        double timestamp = m_pts_info_vec[0].time_stamp;
        if (m_if_color_map)
        {
            double firsttimestamp = m_pts_info_vec[0].time_stamp;
            int img_list_size = m_img_timestamp_vec.size();

            log_file1 << "list size: " << img_list_size << endl;

            for (int i = 0; i < img_list_size; i++)
            {
                if (str2double(m_img_timestamp_vec[i]) > firsttimestamp + m_first_receive_time)
                {
                    idx_select = i;
                    break;
                }
            }

            if (idx_select > 0)
            {
                idx_select = (firsttimestamp + m_first_receive_time > (str2double(m_img_timestamp_vec[idx_select - 1]) / 2.0 + str2double(m_img_timestamp_vec[idx_select]) / 2.0)) ? idx_select : idx_select - 1;
            }
            else
            {
                idx_select = 0;
            }

            cv::Mat img = cv::imread(m_img_path + "img/" + m_img_timestamp_vec[idx_select] + ".png");
            cv::undistort(img, unImg, K, DistortCoeff);
        }

        int corner_num = 0;
        int surface_num = 0;
        int full_num = 0;
        pc_corners.resize(m_pts_info_vec.size());
        pc_surface.resize(m_pts_info_vec.size());
        pc_full_res.resize(m_pts_info_vec.size());
        pc_full_color.resize(m_pts_info_vec.size());
        float maximum_idx = maximum_blur * m_pts_info_vec.size();
        float minimum_idx = minimum_blur * m_pts_info_vec.size();
        int pt_critical_rm_mask = e_pt_000 | e_pt_nan | e_pt_too_near;
        for (size_t i = 0; i < m_pts_info_vec.size(); i++)
        {
            if (m_pts_info_vec[i].idx > maximum_idx ||
                m_pts_info_vec[i].idx < minimum_idx)
                continue;

            if ((m_pts_info_vec[i].pt_type & pt_critical_rm_mask) == 0)
            {
                if (m_pts_info_vec[i].pt_label & e_label_corner) // 角点
                {
                    if (m_pts_info_vec[i].pt_type != e_pt_normal)
                        continue;

                    if (m_pts_info_vec[i].depth_sq2 < std::pow(30, 2))
                    {
                        pc_corners.points[corner_num] = m_raw_pts_vec[i];
                        pc_corners.points[corner_num].intensity = m_pts_info_vec[i].time_stamp;
                        corner_num++;
                    }
                }

                if (m_pts_info_vec[i].pt_label & e_label_surface) // 平面点
                {
                    if (m_pts_info_vec[i].depth_sq2 < std::pow(1000, 2))
                    {
                        pc_surface.points[surface_num] = m_raw_pts_vec[i];
                        pc_surface.points[surface_num].intensity = float(m_pts_info_vec[i].time_stamp);
                        surface_num++;
                    }
                }
            }

            int rgb[3] = {0, 0, 0};
            if (m_if_color_map)
            {
                cv::Mat pos(4, 1, CV_32F);
                pos.at<float>(0, 0) = (float)m_raw_pts_vec[i].x;
                pos.at<float>(1, 0) = (float)m_raw_pts_vec[i].y;
                pos.at<float>(2, 0) = (float)m_raw_pts_vec[i].z;
                pos.at<float>(3, 0) = (float)1.0;

                cv::Mat pixel = K * Tcl * pos;

                int x = cvRound(pixel.at<float>(0, 0) / pixel.at<float>(2, 0));
                int y = cvRound(pixel.at<float>(1, 0) / pixel.at<float>(2, 0));

                if (x < unImg.cols && x >= 0 && y < unImg.rows && y >= 0)
                {
                    rgb[0] = (int)unImg.at<cv::Vec3b>(y, x)[2];
                    rgb[1] = (int)unImg.at<cv::Vec3b>(y, x)[1];
                    rgb[2] = (int)unImg.at<cv::Vec3b>(y, x)[0];
                }

                if (timestamp - 30.499641 < 0.001 && timestamp - 30.499641 > -0.001)
                {
                    log_file2 << m_raw_pts_vec[i].x << "  " << m_raw_pts_vec[i].y << "  " << m_raw_pts_vec[i].z << "            " << x << "  " << y << endl;
                }
            }

            pc_full_color.points[full_num].x = (float)rgb[0];
            pc_full_color.points[full_num].y = (float)rgb[1];
            pc_full_color.points[full_num].z = (float)rgb[2];
            pc_full_res.points[full_num] = m_raw_pts_vec[i];
            pc_full_res.points[full_num].intensity = m_pts_info_vec[i].time_stamp;
            full_num++;
        }

        cv::imwrite("/home/zjucvg/out1.png", unImg);

        pc_corners.resize(corner_num);
        pc_surface.resize(surface_num);
        pc_full_res.resize(full_num);
        pc_full_color.resize(full_num);
    }

    template <typename T>
    void set_intensity(T &pt, const E_intensity_type &i_type = e_I_motion_blur)
    {
        Pt_infos *pt_info = find_pt_info(pt);
        switch (i_type)
        {
        case (e_I_raw):
            pt.intensity = pt_info->raw_intensity;
            break;
        case (e_I_motion_blur):
            pt.intensity = ((float)pt_info->idx) / (float)m_input_points_size;
            assert(pt.intensity <= 1.0 && pt.intensity >= 0.0);
            break;
        case (e_I_motion_mix):
            pt.intensity = 0.1 * ((float)pt_info->idx + 1) / (float)m_input_points_size + (int)(pt_info->raw_intensity);
            break;
        case (e_I_scan_angle):
            pt.intensity = pt_info->polar_angle;
            break;
        case (e_I_curvature):
            pt.intensity = pt_info->curvature;
            break;
        case (e_I_view_angle):
            pt.intensity = pt_info->view_angle;
            break;
        case (e_I_time_stamp):
            pt.intensity = pt_info->time_stamp;
        default:
            pt.intensity = ((float)pt_info->idx + 1) / (float)m_input_points_size;
        }

        return;
    }

    template <typename T>
    cv::Mat draw_dbg_img(cv::Mat &img, std::vector<T> &pt_list_eigen, cv::Scalar color = cv::Scalar::all(255), int radius = 3)
    {
        cv::Mat res_img = img.clone();
        unsigned int pt_size = pt_list_eigen.size();
        for (unsigned int idx = 0; idx < pt_size; idx++)
        {
            draw_pt(res_img, pt_list_eigen[idx], color, radius);
        }

        return res_img;
    }

    void add_mask_of_point(Pt_infos *pt_infos, const E_point_type &pt_type, int neighbor_count = 0)
    {
        int idx = pt_infos->idx;
        pt_infos->pt_type |= pt_type;
        if (neighbor_count > 0)
        {
            for (int i = -neighbor_count; i < neighbor_count; i++)
            {
                idx = pt_infos->idx + i;
                if (i != 0 && (idx >= 0) && (idx < (int)m_pts_info_vec.size())) // not the point itself && not the first point in a sweep && not beyond the max_point_num
                {
                    m_pts_info_vec[idx].pt_type |= pt_type; // bitwise |=, if pt_type is !0(which is 'e_pt_normal'--- not yet processed), then assign a point_type
                }
            }
        }
    }

    void eval_point(Pt_infos *pt_info)
    {
        // 深度值特别小的点
        if (pt_info->depth_sq2 < m_livox_min_allow_dis * m_livox_min_allow_dis)
        {
            add_mask_of_point(pt_info, e_pt_too_near);
        }

        pt_info->sigma = pt_info->raw_intensity / pt_info->polar_dis_sq2;
        if (pt_info->sigma < m_livox_min_sigma)
        {
            add_mask_of_point(pt_info, e_pt_reflectivity_low);
        }
    }

    // 计算曲率和视角
    void compute_features()
    {
        unsigned int pts_size = m_raw_pts_vec.size();
        size_t curvature_ssd_size = 2;               // 邻接数
        int critical_rm_point = e_pt_000 | e_pt_nan; // 按位或运算 --empty | inf
        float neighbor_accumulate_xyz[3] = {0.0, 0.0, 0.0};

        for (size_t idx = curvature_ssd_size; idx < pts_size - curvature_ssd_size; idx++)
        {
            if (m_pts_info_vec[idx].pt_type & critical_rm_point) // 点是否为无效点（empty | inf）
            {
                continue;
            }

            /*********** 计算曲率 ************/
            neighbor_accumulate_xyz[0] = 0.0;
            neighbor_accumulate_xyz[1] = 0.0;
            neighbor_accumulate_xyz[2] = 0.0;

            for (size_t i = 1; i <= curvature_ssd_size; i++)
            {
                if ((m_pts_info_vec[idx + i].pt_type & e_pt_000) || (m_pts_info_vec[idx - i].pt_type & e_pt_000)) // 如果左或右邻近点为空
                {
                    if (i == 1)
                    {
                        m_pts_info_vec[idx].pt_label |= e_label_near_zero;
                    }
                    else
                    {
                        m_pts_info_vec[idx].pt_label = e_label_invalid;
                    }

                    break;
                }
                else if ((m_pts_info_vec[idx + i].pt_type & e_pt_nan) || (m_pts_info_vec[idx - i].pt_type & e_pt_nan)) // 如果左或右邻近点为空 inf
                {
                    if (i == 1)
                    {
                        m_pts_info_vec[idx].pt_label |= e_label_near_nan;
                    }
                    else
                    {
                        m_pts_info_vec[idx].pt_label = e_label_invalid;
                    }

                    break;
                }
                else // 太远或太近
                {
                    neighbor_accumulate_xyz[0] += m_raw_pts_vec[idx + i].x + m_raw_pts_vec[idx - i].x;
                    neighbor_accumulate_xyz[1] += m_raw_pts_vec[idx + i].y + m_raw_pts_vec[idx - i].y;
                    neighbor_accumulate_xyz[2] += m_raw_pts_vec[idx + i].z + m_raw_pts_vec[idx - i].z;
                }
            }

            if (m_pts_info_vec[idx].pt_label == e_label_invalid)
            {
                continue;
            }

            neighbor_accumulate_xyz[0] -= curvature_ssd_size * 2 * m_raw_pts_vec[idx].x;
            neighbor_accumulate_xyz[1] -= curvature_ssd_size * 2 * m_raw_pts_vec[idx].y;
            neighbor_accumulate_xyz[2] -= curvature_ssd_size * 2 * m_raw_pts_vec[idx].z;
            m_pts_info_vec[idx].curvature = neighbor_accumulate_xyz[0] * neighbor_accumulate_xyz[0] + neighbor_accumulate_xyz[1] * neighbor_accumulate_xyz[1] +
                                            neighbor_accumulate_xyz[2] * neighbor_accumulate_xyz[2];

            /*********** 计算平面角 ************/
            Eigen::Matrix<float, 3, 1> vec_a(m_raw_pts_vec[idx].x, m_raw_pts_vec[idx].y, m_raw_pts_vec[idx].z);
            Eigen::Matrix<float, 3, 1> vec_b(m_raw_pts_vec[idx + curvature_ssd_size].x - m_raw_pts_vec[idx - curvature_ssd_size].x,
                                             m_raw_pts_vec[idx + curvature_ssd_size].y - m_raw_pts_vec[idx - curvature_ssd_size].y,
                                             m_raw_pts_vec[idx + curvature_ssd_size].z - m_raw_pts_vec[idx - curvature_ssd_size].z);
            m_pts_info_vec[idx].view_angle = Eigen_math::vector_angle(vec_a, vec_b, 1) * 57.3; // 扫描平面和当前点直接的夹角，乘57.3转换为角度

            if (m_pts_info_vec[idx].view_angle > minimum_view_angle) // 大于10°
            {
                if (m_pts_info_vec[idx].curvature < thr_surface_curvature) // 曲率小于0.01, 认为时平面点
                {
                    m_pts_info_vec[idx].pt_label |= e_label_surface; // 标记为平面点
                }

                float sq2_diff = 0.1;
                if (m_pts_info_vec[idx].curvature > thr_corner_curvature) // 曲率大于0.05，认为时边界点
                {
                    if (m_pts_info_vec[idx].depth_sq2 <= m_pts_info_vec[idx - curvature_ssd_size].depth_sq2 &&
                        m_pts_info_vec[idx].depth_sq2 <= m_pts_info_vec[idx + curvature_ssd_size].depth_sq2)
                    {
                        if (abs(m_pts_info_vec[idx].depth_sq2 - m_pts_info_vec[idx - curvature_ssd_size].depth_sq2) < sq2_diff * m_pts_info_vec[idx].depth_sq2 ||
                            abs(m_pts_info_vec[idx].depth_sq2 - m_pts_info_vec[idx + curvature_ssd_size].depth_sq2) < sq2_diff * m_pts_info_vec[idx].depth_sq2)
                            m_pts_info_vec[idx].pt_label |= e_label_corner;
                    }
                }
            }
        }
    }

    template <typename T>
    int projection_scan_3d_2d(pcl::PointCloud<T> &laserCloudIn, std::vector<float> &scan_id_index)
    {
        unsigned int pts_size = laserCloudIn.size();
        m_pts_info_vec.clear();
        m_pts_info_vec.resize(pts_size);
        m_raw_pts_vec.resize(pts_size);
        std::vector<int> edge_idx;
        std::vector<int> split_idx;
        scan_id_index.resize(pts_size);
        edge_idx.clear();
        m_map_pt_idx.clear();
        m_map_pt_idx.reserve(pts_size);
        std::vector<int> zero_idx;

        m_input_points_size = 0;

        for (unsigned int idx = 0; idx < pts_size; idx++)
        {
            m_raw_pts_vec[idx] = laserCloudIn.points[idx];
            Pt_infos *pt_info = &m_pts_info_vec[idx];                               // livox点属性
            m_map_pt_idx.insert(std::make_pair(laserCloudIn.points[idx], pt_info)); // std::make_pair->std::map< PointType, Pt_infos *, Pt_compare >
            pt_info->raw_intensity = laserCloudIn.points[idx].intensity;
            pt_info->idx = idx;
            pt_info->time_stamp = m_current_time + ((float)idx) * m_time_internal_pts;
            m_last_maximum_time_stamp = pt_info->time_stamp;
            m_input_points_size++;

            log_file1 << pt_info->time_stamp << endl;

            if (!std::isfinite(laserCloudIn.points[idx].x) ||
                !std::isfinite(laserCloudIn.points[idx].y) ||
                !std::isfinite(laserCloudIn.points[idx].z))
            {
                add_mask_of_point(pt_info, e_pt_nan); // 点值为inf
                continue;
            }

            if (laserCloudIn.points[idx].x == 0)
            {
                if (idx == 0)
                {
                    // 第一个点
                    screen_out << "First point should be normal!!!" << std::endl;

                    pt_info->pt_2d_img << 0.01, 0.01; // 第一个点的二维投影坐标
                    pt_info->polar_dis_sq2 = 0.0001;  // x * x + y * y + z * z init to 0.0001
                    add_mask_of_point(pt_info, e_pt_000);
                }
                else
                {
                    pt_info->pt_2d_img = m_pts_info_vec[idx - 1].pt_2d_img;
                    pt_info->polar_dis_sq2 = m_pts_info_vec[idx - 1].polar_dis_sq2;
                    add_mask_of_point(pt_info, e_pt_000);
                    continue;
                }
            }

            m_map_pt_idx.insert(std::make_pair(laserCloudIn.points[idx], pt_info));

            pt_info->depth_sq2 = depth2_xyz(laserCloudIn.points[idx].x, laserCloudIn.points[idx].y, laserCloudIn.points[idx].z); // x * x + y * y + z * z

            pt_info->pt_2d_img << laserCloudIn.points[idx].y / laserCloudIn.points[idx].x, laserCloudIn.points[idx].z / laserCloudIn.points[idx].x; // X==1平面上，3D投影到2D
            pt_info->polar_dis_sq2 = dis2_xy(pt_info->pt_2d_img(0), pt_info->pt_2d_img(1));                                                         // 将点投影到Y-Z二维平面上

            // 评估点坐标
            eval_point(pt_info);
            // 点位于视场角边界上
            if (pt_info->polar_dis_sq2 > m_max_edge_polar_pos) // m_max_edge_polar_pos = std::pow( tan( max_fov / 57.3 ) * 1, 2 )
            {
                add_mask_of_point(pt_info, e_pt_circle_edge, 2); // TODO -2to+2. 5 points as edge point?? 一人越线，全家受累??
            }

            // 分割scans
            if (idx >= 1) // 不是扫描线的第一个点
            {
                // 判断点是靠近花瓣中心，还是远离花瓣中心
                float dis_incre = pt_info->polar_dis_sq2 - m_pts_info_vec[idx - 1].polar_dis_sq2;

                if (dis_incre > 0) // 远离
                {
                    pt_info->polar_direction = 1;
                }

                if (dis_incre < 0) // 靠近
                {
                    pt_info->polar_direction = -1;
                }

                if (pt_info->polar_direction == -1 && m_pts_info_vec[idx - 1].polar_direction == 1) // 检测花瓣上的转折点
                {
                    if (edge_idx.size() == 0 || (idx - split_idx[split_idx.size() - 1]) > 50) // TODO if no edge size lodged or 50 points after the last split, why 50?
                    {
                        split_idx.push_back(idx); //# of petal of scan
                        edge_idx.push_back(idx);  // edge side points
                        continue;
                    }
                }

                if (pt_info->polar_direction == 1 && m_pts_info_vec[idx - 1].polar_direction == -1) // other-way-around turnning point which close to the center.
                {
                    if (zero_idx.size() == 0 || (idx - split_idx[split_idx.size() - 1]) > 50)
                    {
                        split_idx.push_back(idx); //# of petal of scan
                        zero_idx.push_back(idx); //center side points
                        continue;
                    }
                }
            }
        }

        split_idx.push_back(pts_size - 1); // last slot of split_idx record # of points in last callback

        int val_index = 0;
        int pt_angle_index = 0;
        float scan_angle = 0;
        int internal_size = 0;

        if (split_idx.size() < 6)
            return 0;

        for (int idx = 0; idx < (int)pts_size; idx++)
        {
            if (val_index < split_idx.size() - 2) // number of item in split_idx is .size-1, and the last slot is the total number of points recorded. therefore -2
            {
                if (idx == 0 || idx > split_idx[val_index + 1]) // if idx is the staring point or the beginning of a petal
                {
                    if (idx > split_idx[val_index + 1]) // if idx is the beginning of a petal, then petal count++
                    {
                        val_index++;
                    }

                    internal_size = split_idx[val_index + 1] - split_idx[val_index]; // number of points in this petal

                    if (m_pts_info_vec[split_idx[val_index + 1]].polar_dis_sq2 > 10000) // if the starting point of next petal's x^2+y^2+z^2 > 10000
                    {
                        pt_angle_index = split_idx[val_index + 1] - (int)(internal_size * 0.20);                                             // TODO last 20% of the points in this petal
                        scan_angle = atan2(m_pts_info_vec[pt_angle_index].pt_2d_img(1), m_pts_info_vec[pt_angle_index].pt_2d_img(0)) * 57.3; // pt_2d_img(1):z/x,(0):y/x
                        scan_angle = scan_angle + 180.0;
                    }
                    else
                    {
                        pt_angle_index = split_idx[val_index + 1] - (int)(internal_size * 0.80);                                             // TODO last 80% of the points in this petal
                        scan_angle = atan2(m_pts_info_vec[pt_angle_index].pt_2d_img(1), m_pts_info_vec[pt_angle_index].pt_2d_img(0)) * 57.3; // atan2(z/y)*57.3, angle of laser beam in degrees
                        scan_angle = scan_angle + 180.0;
                    }
                }
            }

            m_pts_info_vec[idx].polar_angle = scan_angle; // keep in mind the above, the if conditions skip points, therefore most scan_angle will be 0
            scan_id_index[idx] = scan_angle;              // only scan_angle of the starting point of a petal recorded
        }

        return split_idx.size() - 1;
    }

    void reorder_laser_cloud_scan(std::vector<pcl::PointCloud<pcl::PointXYZI>> &in_laserCloudScans, std::vector<std::vector<int>> &pts_mask)
    {
        unsigned int min_pts_per_scan = 0;
        screen_out << "Before reorder" << endl;

        std::vector<pcl::PointCloud<pcl::PointXYZI>> res_laser_cloud(in_laserCloudScans.size()); // abandon first and last
        std::vector<std::vector<int>> res_pts_mask(in_laserCloudScans.size());
        std::map<float, int> map_angle_idx;

        for (unsigned int i = 0; i < in_laserCloudScans.size() - 0; i++)
        {
            if (in_laserCloudScans[i].size() > min_pts_per_scan)
            {
                map_angle_idx.insert(std::make_pair(in_laserCloudScans[i].points[0].intensity, i));
            }
            else
            {
                continue;
            }
        }

        screen_out << "After reorder" << endl;
        std::map<float, int>::iterator it;
        int current_index = 0;

        for (it = map_angle_idx.begin(); it != map_angle_idx.end(); it++)
        {
            if (in_laserCloudScans[it->second].size() > min_pts_per_scan)
            {
                res_laser_cloud[current_index] = in_laserCloudScans[it->second];
                res_pts_mask[current_index] = pts_mask[it->second];
                current_index++;
            }
        }

        res_laser_cloud.resize(current_index);
        res_pts_mask.resize(current_index);

        in_laserCloudScans = res_laser_cloud;
        pts_mask = res_pts_mask;
        screen_out << "Return size = " << pts_mask.size() << "  " << in_laserCloudScans.size() << endl;

        return;
    }

    // 分割整个点云
    template <typename T>
    void split_laser_scan(const int clutter_size, const pcl::PointCloud<T> &laserCloudIn,
                          const std::vector<float> &scan_id_index,
                          std::vector<pcl::PointCloud<PointType>> &laserCloudScans)
    {
        std::vector<std::vector<int>> pts_mask;
        laserCloudScans.resize(clutter_size);
        pts_mask.resize(clutter_size);

        PointType point;
        int scan_idx = 0;
        for (unsigned int i = 0; i < laserCloudIn.size(); i++)
        {
            point = laserCloudIn.points[i];
            if (i > 0 && ((scan_id_index[i]) != (scan_id_index[i - 1])))
            {
                scan_idx = scan_idx + 1;
                pts_mask[scan_idx].reserve(5000); // 不同型号Livox扫描仪的点数不一样
            }

            laserCloudScans[scan_idx].push_back(point);
            pts_mask[scan_idx].push_back(m_pts_info_vec[i].pt_type);
        }

        laserCloudScans.resize(scan_idx);

        int remove_point_pt_type = e_pt_000 |
                                   e_pt_too_near |
                                   e_pt_nan;
        int scan_avail_num = 0;
        std::vector<pcl::PointCloud<PointType>> res_laser_cloud_scan;
        for (unsigned int i = 0; i < laserCloudScans.size(); i++)
        {
            scan_avail_num = 0;
            pcl::PointCloud<PointType> laser_clour_per_scan;
            for (unsigned int idx = 0; idx < laserCloudScans[i].size(); idx++)
            {
                if ((pts_mask[i][idx] & remove_point_pt_type) == 0)
                {
                    if (laserCloudScans[i].points[idx].x == 0)
                    {
                        screen_printf("Error!!! Mask = %d\r\n", pts_mask[i][idx]);
                        assert(laserCloudScans[i].points[idx].x != 0);
                        continue;
                    }
                    auto temp_pt = laserCloudScans[i].points[idx];
                    set_intensity(temp_pt, default_return_intensity_type); // rviz中显示用到的颜色
                    laser_clour_per_scan.points.push_back(temp_pt);
                    scan_avail_num++;
                }
            }

            if (scan_avail_num)
            {
                res_laser_cloud_scan.push_back(laser_clour_per_scan);
            }
        }

        laserCloudScans = res_laser_cloud_scan;
    }

    // 提取Livox特征点
    template <typename T>
    std::vector<pcl::PointCloud<pcl::PointXYZI>> extract_laser_features(pcl::PointCloud<T> &laserCloudIn, double time_stamp = -1)
    {
        assert(time_stamp >= 0.0);
        if (time_stamp <= 0.0000001 || (time_stamp < m_last_maximum_time_stamp))
        {
            m_current_time = m_last_maximum_time_stamp;
        }
        else
        {
            m_current_time = time_stamp - m_first_receive_time;
        }
        
        if (m_first_receive_time <= 0)
        {
            m_first_receive_time = time_stamp;
        }

        screen_out << "Extract feature, input: " << time_stamp << ", first time: " << m_first_receive_time << ", current_time: " << m_current_time << endl;
        std::vector<pcl::PointCloud<PointType>> laserCloudScans, temp_laser_scans;
        std::vector<float> scan_id_index;
        laserCloudScans.clear();
        m_map_pt_idx.clear();
        m_pts_info_vec.clear();
        m_raw_pts_vec.clear();
        if (m_if_save_pcd_file)
        {
            stringstream ss;
            ss << PCL_DATA_SAVE_DIR << "/pc_" << pcl_data_save_index << ".pcd" << endl;
            pcl_data_save_index = pcl_data_save_index + 1;
            screen_out << "Save file = " << ss.str() << endl;
            pcl::io::savePCDFileASCII(ss.str(), laserCloudIn);
        }

        int clutter_size = projection_scan_3d_2d(laserCloudIn, scan_id_index); // 返回一个scan的花瓣数，并将一个scan的点云分成多个花瓣点云集合
        compute_features();                                                    // 分类为边缘点和平面点
        if (clutter_size == 0)
        {
            return laserCloudScans; // 没有花瓣
        }
        else
        {
            split_laser_scan(clutter_size, laserCloudIn, scan_id_index, laserCloudScans); // 为了RVIZ显示，将点云分成不同的颜色
            return laserCloudScans;
        }
    }
};

#endif // LIVOX_LASER_SCAN_HANDLER_HPP
