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

#ifndef LASER_FEATURE_EXTRACTION_H
#define LASER_FEATURE_EXTRACTION_H

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <vector>

#include "livox_feature_extractor.hpp"
#include "tools/common.h"
#include "tools/tools_logger.hpp"

using std::atan2;
using std::cos;
using std::sin;
using namespace Common_tools;

class Laser_feature
{
  public:
    const double m_para_scanPeriod = 0.1;

    int m_if_pub_debug_feature = 1;

    const int   m_para_system_delay = 20;
    int         m_para_system_init_count = 0;
    bool        m_para_systemInited = false;
    float       m_pc_curvature[ 400000 ];
    int         m_pc_sort_idx[ 400000 ];
    int         m_pc_neighbor_picked[ 400000 ];
    int         m_pc_cloud_label[ 400000 ];
    int         m_if_motion_deblur = 0;
    int         m_odom_mode = 0; //0 = for odom, 1 = for mapping
    float       m_plane_resolution;
    float       m_line_resolution;
    int         m_piecewise_number;

    int m_maximum_input_lidar_pointcloud = 3;
    File_logger m_file_logger;

    bool        m_if_pub_each_line = false;
    int         m_lidar_type = 0; // 0 is velodyne, 1 is livox
    int         m_laser_scan_number = 64; // total size of the Lines, 16/64 for Velodyne, Number of half-a-petal for Livox 
    std::mutex  m_mutex_lock_handler;
    Livox_laser m_livox;
    ros::Time   m_init_timestamp;
    ADD_SCREEN_PRINTF_OUT_METHOD;

    bool comp( int i, int j )
    {
        return ( m_pc_curvature[ i ] < m_pc_curvature[ j ] );
    }

    ros::Publisher              m_pub_laser_pc;
    ros::Publisher              m_pub_pc_sharp_corner;
    ros::Publisher              m_pub_pc_less_sharp_corner;
    ros::Publisher              m_pub_pc_surface_flat;
    ros::Publisher              m_pub_pc_surface_less_flat;
    ros::Publisher              m_pub_pc_removed_pt;
    std::vector<ros::Publisher>                                         m_pub_each_scan;

    std::vector<std::string> m_input_lidar_topic_name_vec;
    std::vector<ros::Subscriber> m_sub_input_laser_cloud_vec;
    std::vector< std::vector<pcl::PointCloud<pcl::PointXYZI> > >   m_map_pointcloud_full_vec_vec;
    std::vector< std::vector<pcl::PointCloud<pcl::PointXYZI> > >   m_map_pointcloud_surface_vec_vec;
    std::vector< std::vector<pcl::PointCloud<pcl::PointXYZI> > >   m_map_pointcloud_corner_vec_vec;
    double m_minimum_range = 0.01;

    ros::Publisher            m_pub_pc_livox_corners, m_pub_pc_livox_surface, m_pub_pc_livox_full;
    sensor_msgs::PointCloud2  temp_out_msg;
    pcl::VoxelGrid<PointType> m_voxel_filter_for_surface;
    pcl::VoxelGrid<PointType> m_voxel_filter_for_corner;

    template<typename T>
    T get_ros_parameter(ros::NodeHandle &nh , const std::string parameter_name, T & parameter, const T default_val)
    {
        nh.param<T>(parameter_name.c_str(), parameter, default_val);
        ENABLE_SCREEN_PRINTF;
        screen_out <<"[Features_extraction_ros_param]: " << parameter_name << " ==> " << parameter << std::endl;
        return parameter;
    }


    int                       init_ros_env()
    {

        ros::NodeHandle nh;
        m_init_timestamp = ros::Time::now();
        init_livox_lidar_para(nh);
        get_ros_parameter<int>(nh, "feature_extraction/scan_line", m_laser_scan_number, 16 );
        get_ros_parameter<float>( nh,"feature_extraction/mapping_plane_resolution", m_plane_resolution, 0.8 );
        get_ros_parameter<float>( nh,"feature_extraction/mapping_line_resolution", m_line_resolution, 0.8 );
        get_ros_parameter<double>(nh, "feature_extraction/minimum_range", m_minimum_range, 0.1 );
        get_ros_parameter<int>( nh,"common/if_motion_deblur", m_if_motion_deblur, 1 );
        get_ros_parameter<int>( nh,"common/piecewise_number", m_piecewise_number, 3 );
        get_ros_parameter<int>(nh, "common/odom_mode", m_odom_mode, 0 );

        double livox_corners, livox_surface, minimum_view_angle;
        get_ros_parameter<double>(nh, "feature_extraction/corner_curvature", livox_corners, 0.05 );
        get_ros_parameter<double>(nh, "feature_extraction/surface_curvature", livox_surface, 0.01 );
        get_ros_parameter<double>(nh, "feature_extraction/minimum_view_angle", minimum_view_angle, 10 );
        string log_save_dir_name;
        get_ros_parameter<std::string>( nh,"log_save_dir", log_save_dir_name, "../" );

        m_livox.thr_corner_curvature = livox_corners;
        m_livox.thr_surface_curvature = livox_surface;
        m_livox.minimum_view_angle = minimum_view_angle;

        //livox.m_livox_min_allow_dis = MINIMUM_RANGE;

        screen_printf( "scan line number %d \n", m_laser_scan_number );

        if ( m_laser_scan_number != 16 && m_laser_scan_number != 64 )
        {
            screen_printf( "only support velodyne with 16 or 64 scan line!" );
            return 0;
        }


        m_file_logger.set_log_dir( log_save_dir_name );
        m_file_logger.init( "scanRegistration.log" );
        m_map_pointcloud_full_vec_vec.resize(m_maximum_input_lidar_pointcloud);
        m_map_pointcloud_surface_vec_vec.resize(m_maximum_input_lidar_pointcloud);
        m_map_pointcloud_corner_vec_vec.resize(m_maximum_input_lidar_pointcloud);

        for(int i = 0 ; i< m_maximum_input_lidar_pointcloud; i++) // NOTE m_maximum_input_lidar_pointcloud hard coded to 1, i is 0 in this case
        {
            m_input_lidar_topic_name_vec.push_back(string("laser_points_").append(std::to_string(i)));
            m_sub_input_laser_cloud_vec.push_back( nh.subscribe<sensor_msgs::PointCloud2>( m_input_lidar_topic_name_vec.back(), 10000, boost::bind( &Laser_feature::laserCloudHandler, this, _1,  m_input_lidar_topic_name_vec.back()) ) );
            m_map_pointcloud_full_vec_vec[i].resize(m_piecewise_number);
            m_map_pointcloud_surface_vec_vec[i].resize(m_piecewise_number);
            m_map_pointcloud_corner_vec_vec[i].resize(m_piecewise_number);
        }
        m_pub_laser_pc = nh.advertise<sensor_msgs::PointCloud2>( "/laser_points_2", 10000 ); // 一帧全部点云
        m_pub_pc_sharp_corner = nh.advertise<sensor_msgs::PointCloud2>( "/laser_cloud_sharp", 10000 );
        m_pub_pc_less_sharp_corner = nh.advertise<sensor_msgs::PointCloud2>( "/laser_cloud_less_sharp", 10000 );
        m_pub_pc_surface_flat = nh.advertise<sensor_msgs::PointCloud2>( "/laser_cloud_flat", 10000 );
        m_pub_pc_surface_less_flat = nh.advertise<sensor_msgs::PointCloud2>( "/laser_cloud_less_flat", 10000 );
        m_pub_pc_removed_pt = nh.advertise<sensor_msgs::PointCloud2>( "/laser_remove_points", 10000 ); // 没用到

        m_pub_pc_livox_corners = nh.advertise<sensor_msgs::PointCloud2>( "/pc2_corners", 10000 ); // livox后端使用
        m_pub_pc_livox_surface = nh.advertise<sensor_msgs::PointCloud2>( "/pc2_surface", 10000 ); // livox后端使用
        m_pub_pc_livox_full = nh.advertise<sensor_msgs::PointCloud2>( "/pc2_full", 10000 ); // livox后端使用

        m_voxel_filter_for_surface.setLeafSize( m_plane_resolution / 2, m_plane_resolution / 2, m_plane_resolution / 2 );
        m_voxel_filter_for_corner.setLeafSize( m_line_resolution, m_line_resolution, m_line_resolution );
        if ( m_if_pub_each_line )
        {
            for ( int i = 0; i < m_laser_scan_number; i++ )
            {
                ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>( "/laser_scanid_" + std::to_string( i ), 100 );
                m_pub_each_scan.push_back( tmp );
            }
        }
        return 0;
    }

    ~Laser_feature(){};
    Laser_feature()
    {
        init_ros_env();
    };

    template <typename PointT>
    void removeClosedPointCloud( const pcl::PointCloud<PointT> &cloud_in,
                                 pcl::PointCloud<PointT> &cloud_out, float thres )
    {
        if ( &cloud_in != &cloud_out )
        {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize( cloud_in.points.size() );
        }

        size_t j = 0;

        for ( size_t i = 0; i < cloud_in.points.size(); ++i )
        {
            if ( cloud_in.points[ i ].x * cloud_in.points[ i ].x + cloud_in.points[ i ].y * cloud_in.points[ i ].y + cloud_in.points[ i ].z * cloud_in.points[ i ].z < thres * thres )
                continue;

            cloud_out.points[ j ] = cloud_in.points[ i ];
            j++;
        }

        if ( j != cloud_in.points.size() )
        {
            cloud_out.points.resize( j );
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>( j );
        cloud_out.is_dense = true;
    }

	// laser callback
    void laserCloudHandler( const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg, const std::string & topic_name )
    {
        std::unique_lock<std::mutex> lock(m_mutex_lock_handler);
        int current_lidar_index = 0;
        for(int i = 0; i< m_maximum_input_lidar_pointcloud; i++)
        {
            if(topic_name.compare(m_input_lidar_topic_name_vec[i]) == 0)
            {
                current_lidar_index = i;
                break;
            }
        }
        assert(current_lidar_index < m_maximum_input_lidar_pointcloud);
        // std::cout <<"Time: " << laserCloudMsg->header.stamp.toSec() << ", name = " << topic_name << ", idx= " << current_lidar_index << std::endl;

        std::vector<pcl::PointCloud<PointType>> laserCloudScans( m_laser_scan_number ); // vector of half-a-petals

        // 最开始先积累20帧数据，然后才处理
        if ( !m_para_systemInited )
        {
            m_para_system_init_count++;

            if ( m_para_system_init_count >= m_para_system_delay )
            {
                m_para_systemInited = true;
            }
            else
                return;
        }

        std::vector<int> scanStartInd( 1000, 0 );
        std::vector<int> scanEndInd( 1000, 0 );

        pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
        pcl::fromROSMsg( *laserCloudMsg, laserCloudIn );
        int raw_pts_num = laserCloudIn.size();

        m_file_logger.printf( " Time: %.5f, num_raw: %d, num_filted: %d\r\n", laserCloudMsg->header.stamp.toSec(), raw_pts_num, laserCloudIn.size() );

        size_t cloudSize = laserCloudIn.points.size();

        if ( m_lidar_type ) // Livox scans
        {

            laserCloudScans = m_livox.extract_laser_features( laserCloudIn, laserCloudMsg->header.stamp.toSec() ); // livox scan into half-petals with edge and plane marked and non-good point removed

            if ( laserCloudScans.size() <= 5 ) // less than 5 scan
            {
                return;
            }

            m_laser_scan_number = laserCloudScans.size() * 1.0;

            scanStartInd.resize( m_laser_scan_number );
            scanEndInd.resize( m_laser_scan_number );
            std::fill( scanStartInd.begin(), scanStartInd.end(), 0 );
            std::fill( scanEndInd.begin(), scanEndInd.end(), 0 );

            if ( m_if_pub_debug_feature ) // only for debugging
            {
                /********************************************
                *    Feature extraction for livox lidar     *
                ********************************************/

                int piece_wise = m_piecewise_number; // devided a scan into 3 pieces to reduce motion blur
                if(m_if_motion_deblur)
                {
                    piece_wise = 1; // if no motion blur, then consider scan as one piece
                }
                vector<float> piece_wise_start( piece_wise );
                vector<float> piece_wise_end( piece_wise );

                for ( int i = 0; i < piece_wise; i++ ) // split a scan into n pieces
                {
                    int start_scans, end_scans;

                    start_scans = int( ( m_laser_scan_number * ( i ) ) / piece_wise ); // start from 0,1/3,2/3 of the total half-a-petals, split all petals in a scan callback into 3
                    end_scans = int( ( m_laser_scan_number * ( i + 1 ) ) / piece_wise ) - 1; // end by 1/3,2/3,3/3 of the total half-a-petals

                    int end_idx = laserCloudScans[ end_scans ].size() - 1; // last point in the ending half-petal of current piece
                    piece_wise_start[ i ] = ( ( float ) m_livox.find_pt_info( laserCloudScans[ start_scans ].points[ 0 ] )->idx ) / m_livox.m_pts_info_vec.size(); // using point info to find the point id in orginal scan, std::map with custom compare search
                    piece_wise_end[ i ] = ( ( float ) m_livox.find_pt_info( laserCloudScans[ end_scans ].points[ end_idx ] )->idx ) / m_livox.m_pts_info_vec.size();
                }

                for ( int i = 0; i < piece_wise; i++ )
                {
                  pcl::PointCloud<PointType>::Ptr livox_corners( new pcl::PointCloud<PointType>() ),
                        livox_surface( new pcl::PointCloud<PointType>() ),
                        livox_full( new pcl::PointCloud<PointType>() );
                    m_livox.get_features( *livox_corners, *livox_surface, *livox_full, piece_wise_start[ i ], piece_wise_end[ i ] );
                    m_map_pointcloud_corner_vec_vec[current_lidar_index][i] = *livox_corners; // NOTE list of LiDARs->list of Pieces->corners in the list of half-Petals
                    m_map_pointcloud_surface_vec_vec[current_lidar_index][i] = *livox_surface; // list of LiDARs->list of Pieces->surfaces in the list of half-Petals
                    m_map_pointcloud_full_vec_vec[current_lidar_index][i] = *livox_full; // list of LiDARs->list of Pieces->all points in the list of half-Petals
                    
                }

                //if(1)

                for ( int i = 0; i < piece_wise; i++ ) // for every piece of the current scan
                {

                    ros::Time                       current_time = ros::Time::now();
                    pcl::PointCloud<PointType>::Ptr livox_corners( new pcl::PointCloud<PointType>() ),
                        livox_surface( new pcl::PointCloud<PointType>() ),
                        livox_full( new pcl::PointCloud<PointType>() );
                    if ( 1 )
                    {
                        if ( current_lidar_index != 0 )
                        {
                            return;
                        }

                        for ( int ii = 0; ii < m_maximum_input_lidar_pointcloud; ii++ ) // m_maximum_input_lidar_pointcloud is 1, number of LiDARs using
                        {
                            *livox_full += m_map_pointcloud_full_vec_vec[ ii ][ i ]; // these lines suppose to gather all features from all LiDARs
                            *livox_surface += m_map_pointcloud_surface_vec_vec[ ii ][ i ];
                            *livox_corners += m_map_pointcloud_corner_vec_vec[ ii ][ i ];
                        }
                    }
                    else // no you saw the 1 up there, here is no else
                    {
                        *livox_full = m_map_pointcloud_full_vec_vec[ current_lidar_index ][ i ];
                        *livox_surface = m_map_pointcloud_surface_vec_vec[ current_lidar_index ][ i ];
                        *livox_corners = m_map_pointcloud_corner_vec_vec[ current_lidar_index ][ i ];
                    }
                    // NOTE packing and sending featured points
                    pcl::toROSMsg( *livox_full, temp_out_msg );
                    temp_out_msg.header.stamp = current_time;
                    temp_out_msg.header.frame_id = "camera_init";
                    m_pub_pc_livox_full.publish( temp_out_msg );

                    m_voxel_filter_for_surface.setInputCloud( livox_surface );
                    m_voxel_filter_for_surface.filter( *livox_surface );
                    pcl::toROSMsg( *livox_surface, temp_out_msg );
                    temp_out_msg.header.stamp = current_time;
                    temp_out_msg.header.frame_id = "camera_init";
                    m_pub_pc_livox_surface.publish( temp_out_msg );

                    m_voxel_filter_for_corner.setInputCloud( livox_corners );
                    m_voxel_filter_for_corner.filter( *livox_corners );
                    pcl::toROSMsg( *livox_corners, temp_out_msg );
                    temp_out_msg.header.stamp = current_time;
                    temp_out_msg.header.frame_id = "camera_init";
                    m_pub_pc_livox_corners.publish( temp_out_msg );
                    if ( m_odom_mode == 0 ) // odometry mode TODO odometry mode时，只使用一段livox激光点云
                    {
                        break;
                    }
                }
            }
            return; // 回调函数结束，后面的只有雷达为velodyne时才执行。
        } // end livox laser
        else // NOTE if using Velodyne, this part is modified from A-LOAM package, with some attributes renamed
        {
            /********************************************
             *    Feature extraction for velodyne lidar *
            ********************************************/
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud( laserCloudIn, laserCloudIn, indices );
            removeClosedPointCloud( laserCloudIn, laserCloudIn, m_minimum_range ); // TODO 好像这样并没有移除最近的points

            //printf_line;
            float startOri = -atan2( laserCloudIn.points[ 0 ].y, laserCloudIn.points[ 0 ].x );
            float endOri = -atan2( laserCloudIn.points[ cloudSize - 1 ].y,
                                   laserCloudIn.points[ cloudSize - 1 ].x ) +
                           2 * M_PI;

            if ( endOri - startOri > 3 * M_PI )
            {
                endOri -= 2 * M_PI;
            }
            else if ( endOri - startOri < M_PI )
            {
                endOri += 2 * M_PI;
            }

            //printf("end Ori %f\n", endOri);

            //printf_line;
            bool      halfPassed = false;
            int       count = cloudSize;
            PointType point;

            for ( size_t i = 0; i < cloudSize; i++ )
            {
                point.x = laserCloudIn.points[ i ].x;
                point.y = laserCloudIn.points[ i ].y;
                point.z = laserCloudIn.points[ i ].z;

                float angle = atan( point.z / sqrt( point.x * point.x + point.y * point.y ) ) * 180 / M_PI;
                int   scanID = 0;

                if ( m_laser_scan_number == 16 )
                {
                    scanID = int( ( angle + 15 ) / 2 + 0.5 );

                    if ( scanID > ( m_laser_scan_number - 1 ) || scanID < 0 )
                    {
                        count--;
                        continue;
                    }
                }
                else if ( m_laser_scan_number == 64 )
                {
                    if ( angle >= -8.83 )
                    {
                        scanID = int( ( 2 - angle ) * 3.0 + 0.5 );
                    }
                    else
                    {
                        scanID = m_laser_scan_number / 2 + int( ( -8.83 - angle ) * 2.0 + 0.5 );
                    }

                    // use [0 50]  > 50 remove outlies
                    if ( angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0 )
                    {
                        count--;
                        continue;
                    }
                }
                else
                {
                    printf( "wrong scan number\n" );
                    ROS_BREAK();
                }

                //printf("angle %f scanID %d \n", angle, scanID);
                //printf_line;
                float ori = -atan2( point.y, point.x );

                if ( !halfPassed )
                {
                    if ( ori < startOri - M_PI / 2 )
                    {
                        ori += 2 * M_PI;
                    }
                    else if ( ori > startOri + M_PI * 3 / 2 )
                    {
                        ori -= 2 * M_PI;
                    }

                    if ( ori - startOri > M_PI )
                    {
                        halfPassed = true;
                    }
                }
                else
                {
                    ori += 2 * M_PI;

                    if ( ori < endOri - M_PI * 3 / 2 )
                    {
                        ori += 2 * M_PI;
                    }
                    else if ( ori > endOri + M_PI / 2 )
                    {
                        ori -= 2 * M_PI;
                    }
                }

                float relTime = ( ori - startOri ) / ( endOri - startOri );
                point.intensity = scanID + m_para_scanPeriod * relTime;
                laserCloudScans[ scanID ].push_back( point );
            }

            //printf_line;
            //printf( "points size %d \n", cloudSize );
        } // end velodyne laser
        // finished conor/surface/all points
        pcl::PointCloud<PointType>::Ptr laserCloud( new pcl::PointCloud<PointType>() );
        laserCloud->clear();
        cloudSize = 0;
        for ( int i = 0; i < m_laser_scan_number; i++ ) // NOTE indexing all half-a-petal into line of scans, with starting 5 and endding 5 points ignored. m_laser_scan_number = total number of petals
        {
            scanStartInd[ i ] = laserCloud->size() + 5; // scanStartInd is a vector same length as m_laser_scan_number with 0 filled at this point
            //*laserCloud += laserCloudScans[ laserCloudScans.size() - N_SCANS + i ];
            *laserCloud += laserCloudScans[ i ]; // #i half-a-petal
            scanEndInd[ i ] = laserCloud->size() - 6;
            //cloudSize += laserCloudScans[ laserCloudScans.size() - N_SCANS + i ].size();
            cloudSize += laserCloudScans[ i ].size(); // recording total number of points in all petals
        }
        //printf_line;

        for ( size_t i = 5; i < cloudSize - 5; i++ )
        { // NOTE this section calculate the curvature of a point in scanning line(half-a-petal), noting that points already assigned into surface and conners in livox_feature_extractor
            // assert( laserCloud->points[ i - 5 ].x != 0 );
            float diffX = laserCloud->points[ i - 5 ].x + laserCloud->points[ i - 4 ].x + laserCloud->points[ i - 3 ].x + laserCloud->points[ i - 2 ].x + laserCloud->points[ i - 1 ].x - 10 * laserCloud->points[ i ].x + laserCloud->points[ i + 1 ].x + laserCloud->points[ i + 2 ].x + laserCloud->points[ i + 3 ].x + laserCloud->points[ i + 4 ].x + laserCloud->points[ i + 5 ].x;
            float diffY = laserCloud->points[ i - 5 ].y + laserCloud->points[ i - 4 ].y + laserCloud->points[ i - 3 ].y + laserCloud->points[ i - 2 ].y + laserCloud->points[ i - 1 ].y - 10 * laserCloud->points[ i ].y + laserCloud->points[ i + 1 ].y + laserCloud->points[ i + 2 ].y + laserCloud->points[ i + 3 ].y + laserCloud->points[ i + 4 ].y + laserCloud->points[ i + 5 ].y;
            float diffZ = laserCloud->points[ i - 5 ].z + laserCloud->points[ i - 4 ].z + laserCloud->points[ i - 3 ].z + laserCloud->points[ i - 2 ].z + laserCloud->points[ i - 1 ].z - 10 * laserCloud->points[ i ].z + laserCloud->points[ i + 1 ].z + laserCloud->points[ i + 2 ].z + laserCloud->points[ i + 3 ].z + laserCloud->points[ i + 4 ].z + laserCloud->points[ i + 5 ].z;
            float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;
            m_pc_curvature[ i ] = diff; // 每个点的曲率
            m_pc_sort_idx[ i ] = i; // 每个点在点云中的索引
            m_pc_neighbor_picked[ i ] = 0;
            m_pc_cloud_label[ i ] = 0;
            if ( 1 ) // this section removing most points if it looks not interested
            { // the more 'if(1)' you see here, the more author suffered during his testing and tuning
                if ( 1 )
                {
                    if ( diff > 0.1 ) // if curvature >0.1
                    {
                        float depth1 = sqrt( laserCloud->points[ i ].x * laserCloud->points[ i ].x +
                                             laserCloud->points[ i ].y * laserCloud->points[ i ].y +
                                             laserCloud->points[ i ].z * laserCloud->points[ i ].z );
                        float depth2 = sqrt( laserCloud->points[ i + 1 ].x * laserCloud->points[ i + 1 ].x +
                                             laserCloud->points[ i + 1 ].y * laserCloud->points[ i + 1 ].y +
                                             laserCloud->points[ i + 1 ].z * laserCloud->points[ i + 1 ].z );

                        if ( depth1 > depth2 ) // if the current point is futher away from lidar than the next point--- the scanning trend in going twards the center of LiDAR
                        {
                            diffX = laserCloud->points[ i + 1 ].x - laserCloud->points[ i ].x * depth2 / depth1;
                            diffY = laserCloud->points[ i + 1 ].y - laserCloud->points[ i ].y * depth2 / depth1;
                            diffZ = laserCloud->points[ i + 1 ].z - laserCloud->points[ i ].z * depth2 / depth1;

                            if ( sqrt( diffX * diffX + diffY * diffY + diffZ * diffZ ) / depth2 < 0.1 )
                            {
                                m_pc_neighbor_picked[ i - 5 ] = 1;
                                m_pc_neighbor_picked[ i - 4 ] = 1;
                                m_pc_neighbor_picked[ i - 3 ] = 1;
                                m_pc_neighbor_picked[ i - 2 ] = 1;
                                m_pc_neighbor_picked[ i - 1 ] = 1;
                                m_pc_neighbor_picked[ i ] = 1;
                            }
                        }
                        else // if the next point is futher from lidar than the current point
                        {
                            diffX = laserCloud->points[ i + 1 ].x * depth1 / depth2 - laserCloud->points[ i ].x;
                            diffY = laserCloud->points[ i + 1 ].y * depth1 / depth2 - laserCloud->points[ i ].y;
                            diffZ = laserCloud->points[ i + 1 ].z * depth1 / depth2 - laserCloud->points[ i ].z;

                            if ( sqrt( diffX * diffX + diffY * diffY + diffZ * diffZ ) / depth1 < 0.1 )
                            {
                                m_pc_neighbor_picked[ i + 1 ] = 1;
                                m_pc_neighbor_picked[ i + 2 ] = 1;
                                m_pc_neighbor_picked[ i + 3 ] = 1;
                                m_pc_neighbor_picked[ i + 4 ] = 1;
                                m_pc_neighbor_picked[ i + 5 ] = 1;
                                m_pc_neighbor_picked[ i + 6 ] = 1;
                            }
                        }
                    }
                }

                if ( 1 )
                {
                    float diffX2 = laserCloud->points[ i ].x - laserCloud->points[ i - 1 ].x;
                    float diffY2 = laserCloud->points[ i ].y - laserCloud->points[ i - 1 ].y;
                    float diffZ2 = laserCloud->points[ i ].z - laserCloud->points[ i - 1 ].z;
                    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

                    float dis = laserCloud->points[ i ].x * laserCloud->points[ i ].x + laserCloud->points[ i ].y * laserCloud->points[ i ].y + laserCloud->points[ i ].z * laserCloud->points[ i ].z;

                    if ( diff > 0.0002 * dis && diff2 > 0.0002 * dis )
                    {
                        m_pc_neighbor_picked[ i ] = 1;
                    }
                }
            } // all these removed points from following processes
        }
//printf_line;

#if !IF_LIVOX_HANDLER_REMOVE
        if ( m_lidar_type != 0 ) // livox已经在前面返回
        //if(1)
        {
            Livox_laser::Pt_infos *pt_info;
            for ( unsigned int idx = 0; idx < cloudSize; idx++ )
            {
                //printf( "Idx = %d, size = %d, pt = [%f,%f,%f]\r\n", idx, cloudSize, laserCloud->points[ idx ].x, laserCloud->points[ idx ].y, laserCloud->points[ idx ].z );
                // printf( "Idx = %d, pt = [%f,%f,%f]\r\n", idx, 1.0,2.0,3.0 );
                pt_info = m_livox.find_pt_info( laserCloud->points[ idx ] );

                if ( pt_info->pt_type != Livox_laser::e_pt_normal ) // if the point is not a normal state
                {
                    //screen_out << "Reject, id = "<<idx << " ---, type = " << livox.m_mask_pointtype[idx] <<std::endl;
                    m_pc_neighbor_picked[ idx ] = 1; // then the point marked as picked---excluded from below processes
                }
            }
        }
//printf_line;
#endif

        //printf_line;

        pcl::PointCloud<PointType> cornerPointsSharp;
        pcl::PointCloud<PointType> cornerPointsLessSharp;
        pcl::PointCloud<PointType> surfPointsFlat;
        pcl::PointCloud<PointType> surfPointsLessFlat; // this one is not used in the code, all the points not selected or removed by the process are consider as lesslflat
        float                      sharp_point_threshold = 0.05;

        //extract corners points and surface points
        for ( int i = 0; i < m_laser_scan_number; i++ ) // total size of the half-a-petal
        {
            pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan( new pcl::PointCloud<PointType> );
            // To ensure the distribution of features point, spilt each scan into 6 parts equally according to their curvature.
            for ( int j = 0; j < 6; j++ )
            { // scanStartInd: incermental index of starting point of a half-a-petal inside a scan
                //Starting of each sub-scan.
                int sp = ( scanStartInd[ i ] * ( 6 - j ) + scanEndInd[ i ] * j ) / 6;
                //Ending of each sub-scan.
                int ep = ( scanStartInd[ i ] * ( 5 - j ) + scanEndInd[ i ] * ( j + 1 ) ) / 6 - 1;

                //sort curvature
                for ( int k = sp + 1; k <= ep; k++ )
                {
                    for ( int l = k; l >= sp + 1; l-- )
                    {
                        if ( m_pc_curvature[ m_pc_sort_idx[ l ] ] < m_pc_curvature[ m_pc_sort_idx[ l - 1 ] ] ) // NOTE bubble sorting big->small curvature
                        {
                            int temp = m_pc_sort_idx[ l - 1 ];
                            m_pc_sort_idx[ l - 1 ] = m_pc_sort_idx[ l ];
                            m_pc_sort_idx[ l ] = temp;
                        }
                    }
                }

                //select the most sharp and flat point
                int largestPickedNum = 0;
                for ( int k = ep; k >= sp; k-- )
                {
                    int ind = m_pc_sort_idx[ k ]; //The index of biggest curvature.

                    if ( m_pc_neighbor_picked[ ind ] == 0 && // if the point is not been picked yet
                         m_pc_curvature[ ind ] > sharp_point_threshold * 10 )
                    {

                        largestPickedNum++;
                        if ( largestPickedNum <= 20 )
                        {
                            m_pc_cloud_label[ ind ] = 2; //2 -> the label sharpest points.
                            cornerPointsSharp.push_back( laserCloud->points[ ind ] );
                            cornerPointsLessSharp.push_back( laserCloud->points[ ind ] );
                        }
                        else if ( largestPickedNum <= 200 )
                        {
                            m_pc_cloud_label[ ind ] = 1; //1 -> the label of less sharpest points.
                            cornerPointsLessSharp.push_back( laserCloud->points[ ind ] );
                        }
                        else
                        {
                            break;
                        }

                        m_pc_neighbor_picked[ ind ] = 1;

                        float times = 100;
                        // delete 5 neighbor of sharpest points. keep the spareness of the cloud
                        for ( int l = 1; l <= 5 * times; l++ )
                        {
                            float diffX = laserCloud->points[ ind + l ].x - laserCloud->points[ ind + l - 1 ].x;
                            float diffY = laserCloud->points[ ind + l ].y - laserCloud->points[ ind + l - 1 ].y;
                            float diffZ = laserCloud->points[ ind + l ].z - laserCloud->points[ ind + l - 1 ].z;
                            if ( diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05 )
                            {
                                break;
                            }

                            m_pc_neighbor_picked[ ind + l ] = 1;
                        }
                        for ( int l = -1; l >= -5 * times; l-- )
                        {
                            float diffX = laserCloud->points[ ind + l ].x - laserCloud->points[ ind + l + 1 ].x;
                            float diffY = laserCloud->points[ ind + l ].y - laserCloud->points[ ind + l + 1 ].y;
                            float diffZ = laserCloud->points[ ind + l ].z - laserCloud->points[ ind + l + 1 ].z;
                            if ( diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05 )
                            {
                                break;
                            }

                            m_pc_neighbor_picked[ ind + l ] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for ( int k = sp; k <= ep; k++ )
                {
                    int ind = m_pc_sort_idx[ k ];

                    if ( m_pc_neighbor_picked[ ind ] == 0 &&
                         m_pc_curvature[ ind ] < sharp_point_threshold )
                    {

                        m_pc_cloud_label[ ind ] = -1; // -1 the lable of flat points
                        surfPointsFlat.push_back( laserCloud->points[ ind ] );

                        smallestPickedNum++;
                        if ( smallestPickedNum >= 5 )
                        { // 0 the label of less flat points.
                            break;
                        }

                        m_pc_neighbor_picked[ ind ] = 1;
                        for ( int l = 1; l <= 5; l++ ) // keep the spareness of the cloud
                        {
                            float diffX = laserCloud->points[ ind + l ].x - laserCloud->points[ ind + l - 1 ].x;
                            float diffY = laserCloud->points[ ind + l ].y - laserCloud->points[ ind + l - 1 ].y;
                            float diffZ = laserCloud->points[ ind + l ].z - laserCloud->points[ ind + l - 1 ].z;
                            if ( diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05 )
                            {
                                break;
                            }

                            m_pc_neighbor_picked[ ind + l ] = 1;
                        }
                        for ( int l = -1; l >= -5; l-- )
                        {
                            float diffX = laserCloud->points[ ind + l ].x - laserCloud->points[ ind + l + 1 ].x;
                            float diffY = laserCloud->points[ ind + l ].y - laserCloud->points[ ind + l + 1 ].y;
                            float diffZ = laserCloud->points[ ind + l ].z - laserCloud->points[ ind + l + 1 ].z;
                            if ( diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05 )
                            {
                                break;
                            }

                            m_pc_neighbor_picked[ ind + l ] = 1;
                        }
                    }
                }

                // The ublabeled point is the less flat points
                for ( int k = sp; k <= ep; k++ )
                {
                    if ( m_pc_cloud_label[ k ] <= 0 )
                    {
                        surfPointsLessFlatScan->push_back( laserCloud->points[ k ] );
                    }
                }
            } // end 6 sub lasers of each line

            // voxel filter for less sharp points.
            pcl::PointCloud<PointType> surfPointsLessFlatScanDS;

            m_voxel_filter_for_surface.setInputCloud( surfPointsLessFlatScan );
            m_voxel_filter_for_surface.filter( surfPointsLessFlatScanDS );

            surfPointsLessFlat += surfPointsLessFlatScanDS;
        } // end all lines

        //printf_line;
        //printf_line;
        //printf( "sort q time %f \n", t_q_sort );
        //printf_line;
        sensor_msgs::PointCloud2 laserCloudOutMsg;
        pcl::toROSMsg( *laserCloud, laserCloudOutMsg );
        laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
        laserCloudOutMsg.header.frame_id = "camera_init";
        m_pub_laser_pc.publish( laserCloudOutMsg );

        sensor_msgs::PointCloud2 cornerPointsSharpMsg;
        pcl::toROSMsg( cornerPointsSharp, cornerPointsSharpMsg );
        cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
        cornerPointsSharpMsg.header.frame_id = "camera_init";
        m_pub_pc_sharp_corner.publish( cornerPointsSharpMsg );

        sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
        pcl::toROSMsg( cornerPointsLessSharp, cornerPointsLessSharpMsg );
        cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
        cornerPointsLessSharpMsg.header.frame_id = "camera_init";
        m_pub_pc_less_sharp_corner.publish( cornerPointsLessSharpMsg );

        sensor_msgs::PointCloud2 surfPointsFlat2;
        pcl::toROSMsg( surfPointsFlat, surfPointsFlat2 );
        surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
        surfPointsFlat2.header.frame_id = "camera_init";
        m_pub_pc_surface_flat.publish( surfPointsFlat2 );

        sensor_msgs::PointCloud2 surfPointsLessFlat2;
        pcl::toROSMsg( surfPointsLessFlat, surfPointsLessFlat2 );
        surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
        surfPointsLessFlat2.header.frame_id = "camera_init";
        m_pub_pc_surface_less_flat.publish( surfPointsLessFlat2 );

        // pub each scam
        if ( m_if_pub_each_line )
        {
            for ( int i = 0; i < m_laser_scan_number; i++ )
            {
                sensor_msgs::PointCloud2 scanMsg;
                pcl::toROSMsg( laserCloudScans[ i ], scanMsg );
                scanMsg.header.stamp = laserCloudMsg->header.stamp;
                scanMsg.header.frame_id = "camera_init";
                m_pub_each_scan[ i ].publish( scanMsg );
            }
        }
    }

    void init_livox_lidar_para( ros::NodeHandle & nh )
    {
        ENABLE_SCREEN_PRINTF;
        string lidar_tpye_name;
        screen_out << "~~~~~ Init livox lidar parameters ~~~~~" << std::endl;
        if(get_ros_parameter<std::string>( nh, "common/lidar_type", lidar_tpye_name , std::string( "" ) ).length() )
        {
            screen_printf( "***** I get lidar_type declaration, lidar_type_name = %s ***** \r\n", lidar_tpye_name.c_str() );

            if ( lidar_tpye_name.compare( "livox" ) == 0 )
            {
                m_lidar_type = 1;
                screen_out << "Set lidar type = livox" << std::endl;
            }
            else
            {
                screen_out << "Set lidar type = velodyne" << std::endl;
                m_lidar_type = 0;
            }
        }
        else
        {
            screen_printf( "***** No lidar_type declaration ***** \r\n" );
            m_lidar_type = 0;
            screen_out << "Set lidar type = velodyne" << std::endl;
        }

        if ( get_ros_parameter<float>( nh, "feature_extraction/livox_min_dis", m_livox.m_livox_min_allow_dis , 0.1 ) )
        {
            screen_out << "Set livox lidar minimum distance= " << m_livox.m_livox_min_allow_dis << std::endl;
        }

        if ( get_ros_parameter<float>( nh, "feature_extraction/livox_min_sigma", m_livox.m_livox_min_sigma, 7e-4 ) )
        {
            screen_out << "Set livox lidar minimum sigama =  " << m_livox.m_livox_min_sigma << std::endl;
        }
        screen_out << "~~~~~ End ~~~~~" << endl;
    }
};

#endif // LASER_FEATURE_EXTRACTION_H
