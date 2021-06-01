// Author: Jiarong Lin          ziv.lin.ljr@gmail.com

#ifndef POINT_CLOUD_REGISTRATION_HPP
#define POINT_CLOUD_REGISTRATION_HPP
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <mutex>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <future>

#include "ceres_icp.hpp"
#include "common.h"
#include "tools_logger.hpp"
#include "pcl_tools.hpp"
#include "tools_timer.hpp"
#include "cell_map_keyframe.hpp"

#define CORNER_MIN_MAP_NUM 0
#define SURFACE_MIN_MAP_NUM 50

#define BLUR_SCALE 1.0

using namespace PCL_TOOLS;
using namespace Common_tools;
//class Laser_mapping;
int g_export_full_count = 0;

class Point_cloud_registration
{
public:
    ceres::LinearSolverType slover_type = ceres::DENSE_SCHUR; // SPARSE_NORMAL_CHOLESKY | DENSE_QR | DENSE_SCHUR

    int line_search_num = 5;
    int IF_LINE_FEATURE_CHECK = 0;
    int plane_search_num = 5;
    int IF_PLANE_FEATURE_CHECK = 0;
    int ICP_PLANE = 1;
    int ICP_LINE = 1;                                            // loop clousre(scene_alignment.hpp)时： 0
    double m_para_buffer_RT[7] = {0, 0, 0, 1, 0, 0, 0};          // 没用到
    double m_para_buffer_RT_last[7] = {0, 0, 0, 1, 0, 0, 0};     // 没用到
    double m_para_buffer_incremental[7] = {0, 0, 0, 1, 0, 0, 0}; // 实部在后

    Eigen::Map<Eigen::Quaterniond> m_q_w_incre = Eigen::Map<Eigen::Quaterniond>(m_para_buffer_incremental);
    Eigen::Map<Eigen::Vector3d> m_t_w_incre = Eigen::Map<Eigen::Vector3d>(m_para_buffer_incremental + 4);
    // incre 和 m_para_buffer_incremental 共用同一个地址，其中一个改变，另一个也被改变
    // 从上一帧end到当前帧end的TF

    double m_interpolatation_theta; // 从上一帧end到当前帧end转过的角度（轴角形式表示时）

    int m_if_motion_deblur = 0; // motion deblur(去模糊), 0: piesewise processing;  1: linear interpolation； set by yaml

    double m_angular_diff = 0;
    double m_t_diff = 0;
    double m_maximum_dis_plane_for_match = 50.0;
    double m_maximum_dis_line_for_match = 2.0;
    Eigen::Matrix<double, 3, 1> m_interpolatation_omega;         // 从上一帧end到当前帧end单位方向向量
    Eigen::Matrix<double, 3, 3> m_interpolatation_omega_hat;     // 从上一帧end到当前帧end单位方向向量的反对称矩阵
    Eigen::Matrix<double, 3, 3> m_interpolatation_omega_hat_sq2; // 从上一帧end到当前帧end单位方向向量的反对称矩阵平方

    const Eigen::Quaterniond m_q_I = Eigen::Quaterniond(1, 0, 0, 0);

    pcl::KdTreeFLANN<PointType> m_kdtree_corner_from_map;
    pcl::KdTreeFLANN<PointType> m_kdtree_surf_from_map;

    Eigen::Quaterniond m_q_w_curr, m_q_w_last;
    Eigen::Vector3d m_t_w_curr, m_t_w_last;

    Common_tools::File_logger *m_logger_common;
    Common_tools::File_logger *m_logger_pcd;
    Common_tools::File_logger *m_logger_timer;

    Common_tools::Timer *m_timer;
    int m_current_frame_index;
    int m_mapping_init_accumulate_frames = 100;
    float m_last_time_stamp = 0;                  // 上一帧点中最大的时间戳(上一帧end)
    float m_para_max_angular_rate = 200.0 / 50.0; // max angular rate = 90.0 /50.0 deg/s
    float m_para_max_speed = 100.0 / 50.0;        // max speed = 10 m/s
    float m_max_final_cost = 100.0;
    int m_para_icp_max_iterations = 20;
    int m_para_cere_max_iterations = 100;
    int m_para_cere_prerun_times = 2;
    float m_minimum_pt_time_stamp = 0;   // 上一帧点中最大的时间戳(上一帧end) ; loop clousre(scene_alignment.hpp)时： 0
    float m_maximum_pt_time_stamp = 1.0; // 本帧点中最大的时间戳(本帧end);     loop clousre(scene_alignment.hpp)时： 1
    double m_minimum_icp_R_diff = 0.01;
    double m_minimum_icp_T_diff = 0.01;

    double m_inliner_dis = 0.02;
    double m_inlier_ratio = 0.80;

    double m_inlier_threshold;
    ceres::Solver::Summary summary;
    ceres::Solver::Summary m_final_opt_summary;
    int m_maximum_allow_residual_block = 1e5;
    Common_tools::Random_generator_float<float> m_rand_float;

    ~Point_cloud_registration(){ };

    Point_cloud_registration()
    {
        m_q_w_last.setIdentity();
        m_t_w_last.setZero();
        m_q_w_curr.setIdentity();
        m_t_w_curr.setZero();
        m_if_verbose_screen_printf = 1;
    };

    ADD_SCREEN_PRINTF_OUT_METHOD;

    void reset_incremtal_parameter()
    {
        m_interpolatation_theta = 0;
        m_interpolatation_omega_hat.setZero();
        m_interpolatation_omega_hat_sq2.setZero();
    };

    /**
 * @param in_blur:  当前帧某个点的时间戳
 * @param min_blur：上一帧点中最大的时间戳
 * @param max_blur：本帧点中最大的时间戳 
 * @return float： 分段去模糊时，返回1；
 *                线性插值去模糊时，返回与当前帧最小时间戳的差占当前整个时间戳的比例
 */
    float refine_blur(float in_blur, const float &min_blur, const float &max_blur)
    {
        float res = 1.0;
        if (m_if_motion_deblur) // 线性插值去模糊； <2> loop clousre(scene_alignment.hpp)时： 0
        {
            res = (in_blur - min_blur) / (max_blur - min_blur);
            if (!std::isfinite(res) || res > 1.0)
                return 1.0;
            else
                return res;
        }

        return res;
    };

    void set_ceres_solver_bound(ceres::Problem &problem, double *para_buffer_RT)
    {
        for (unsigned int i = 0; i < 3; i++)
        {
            problem.SetParameterLowerBound(para_buffer_RT + 4, i, -m_para_max_speed);
            problem.SetParameterUpperBound(para_buffer_RT + 4, i, +m_para_max_speed);
        }
    };

    double compute_inlier_residual_threshold(std::vector<double> residuals, double ratio)
    {
        std::set<double> dis_vec; // 升序
        for (size_t i = 0; i < (size_t)(residuals.size() / 3); i++)
        {
            dis_vec.insert(fabs(residuals[3 * i + 0]) + fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2]));
        }

        return *(std::next(dis_vec.begin(), (int)((ratio)*dis_vec.size())));
    }

    // <1>计算位姿时(laser_mapping.hpp), 计算的是map ---> curr位姿
    // <2>loop clousre(scene_alignment.hpp)，计算的是history --->curr(TF in map)
    int find_out_incremental_transfrom(pcl::PointCloud<PointType>::Ptr in_laser_cloud_corner_from_map, // <1>:local map corners          <2>curr finished keyframe corners(points in map)
                                       pcl::PointCloud<PointType>::Ptr in_laser_cloud_surf_from_map,   // <1>:local map plannes          <2>curr finished keyframe plannes(points in map)
                                       pcl::KdTreeFLANN<PointType> &kdtree_corner_from_map,            // <1>: local map corners kdtree  <2>curr finished keyframe corners kdtree
                                       pcl::KdTreeFLANN<PointType> &kdtree_surf_from_map,              // <1>: local map plannes kdtree  <2>curr finished keyframe plannes kdtree
                                       pcl::PointCloud<PointType>::Ptr laserCloudCornerStack,          // <1>：curr帧corners              <2>history finished keyframe corners(points in map)
                                       pcl::PointCloud<PointType>::Ptr laserCloudSurfStack)            // <1>：curr帧plannes              <2>history finished keyframe plannes(points in map)
    {
        Eigen::Map<Eigen::Quaterniond> q_w_incre = Eigen::Map<Eigen::Quaterniond>(m_para_buffer_incremental);
        Eigen::Map<Eigen::Vector3d> t_w_incre = Eigen::Map<Eigen::Vector3d>(m_para_buffer_incremental + 4);
        // <1>: 用上一帧end到当前帧end TF 作为初值
        // <2>: incre_q = I, incre_t = T， T：curr_finished_keyframe_center - history_finished_keyframe_center
        // <2>: curr_q = I,  curr_t = T   在scene_alignment.hpp中设置的初值
        // <2>: last_q = I,  last_t = 0

        m_kdtree_corner_from_map = kdtree_corner_from_map;
        m_kdtree_surf_from_map = kdtree_surf_from_map;

        pcl::PointCloud<PointType> laser_cloud_corner_from_map = *in_laser_cloud_corner_from_map;
        pcl::PointCloud<PointType> laser_cloud_surf_from_map = *in_laser_cloud_surf_from_map;

        int laserCloudCornerFromMapNum = laser_cloud_corner_from_map.points.size();
        int laserCloudSurfFromMapNum = laser_cloud_surf_from_map.points.size();
        int laser_corner_pt_num = laserCloudCornerStack->points.size();
        int laser_surface_pt_num = laserCloudSurfStack->points.size();

        *(m_logger_timer->get_ostream()) << m_timer->toc_string("Query points for match") << std::endl;
        m_timer->tic("Pose optimization");

        int surf_avail_num = 0;
        int corner_avail_num = 0;
        float minimize_cost = summary.final_cost;
        PointType pointOri, pointSel;
        int corner_rejection_num = 0;
        int surface_rejecetion_num = 0;
        int if_undistore_in_matching = 1;

        // 最开始的前50帧数据，poses默认m_q_w_curr = I, m_t_w_curr = 0
        // 之后每帧的位姿都是通过与local map 匹配得到
        if (laserCloudCornerFromMapNum > CORNER_MIN_MAP_NUM && laserCloudSurfFromMapNum > SURFACE_MIN_MAP_NUM && m_current_frame_index > m_mapping_init_accumulate_frames)
        {
            m_timer->tic("Build kdtree");

            *(m_logger_timer->get_ostream()) << m_timer->toc_string("Build kdtree") << std::endl;
            Eigen::Quaterniond q_last_optimize(1.f, 0.f, 0.f, 0.f);
            Eigen::Vector3d t_last_optimize(0.f, 0.f, 0.f);
            int iterCount = 0;

            std::vector<int> m_point_search_Idx;
            std::vector<float> m_point_search_sq_dis;

            for (iterCount = 0; iterCount < m_para_icp_max_iterations; iterCount++) // 计算位姿时, yaml: 15; loop closure, yaml: 5
            {
                m_point_search_Idx.clear();
                m_point_search_sq_dis.clear();
                corner_avail_num = 0;
                surf_avail_num = 0;
                corner_rejection_num = 0;
                surface_rejecetion_num = 0;

                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
                ceres::Problem::Options problem_options;
                ceres::ResidualBlockId block_id;
                ceres::Problem problem(problem_options);
                std::vector<ceres::ResidualBlockId> residual_block_ids;

                problem.AddParameterBlock(m_para_buffer_incremental, 4, q_parameterization);
                problem.AddParameterBlock(m_para_buffer_incremental + 4, 3);

                // loop clousre(scene_alignment.hpp)时：不用corners计算loop closure，只用plannes计算
                for (int i = 0; i < laser_corner_pt_num; i++) // curr帧corners
                {
                    if (laser_corner_pt_num > 2 * m_maximum_allow_residual_block) // 计算位姿时, yaml: 2*200; loop closure, yaml:3000
                    {
                        if (m_rand_float.rand_uniform() * laser_corner_pt_num > 2 * m_maximum_allow_residual_block)
                        {
                            continue; // [0, 1]之间的随机数
                        }
                    }

                    pointOri = laserCloudCornerStack->points[i];
                    if ((!std::isfinite(pointOri.x)) ||
                        (!std::isfinite(pointOri.y)) ||
                        (!std::isfinite(pointOri.z)))
                        continue;

                    pointAssociateToMap(&pointOri, &pointSel, refine_blur(pointOri.intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp), if_undistore_in_matching);
                    // 用上一帧时刻的位姿作为初值，把当前帧下的点转化到map下

                    if (m_kdtree_corner_from_map.nearestKSearch(pointSel, line_search_num, m_point_search_Idx, m_point_search_sq_dis) != line_search_num)
                    {
                        continue;
                    }

                    // 在local map 中找离pointSel最近的5个points, 距离由小到大

                    if (m_point_search_sq_dis[line_search_num - 1] < m_maximum_dis_line_for_match)
                    {
                        bool line_is_avail = true;
                        std::vector<Eigen::Vector3d> nearCorners;
                        Eigen::Vector3d center(0, 0, 0);
                        if (IF_LINE_FEATURE_CHECK)
                        {
                            for (int j = 0; j < line_search_num; j++)
                            {
                                Eigen::Vector3d tmp(laser_cloud_corner_from_map.points[m_point_search_Idx[j]].x,
                                                    laser_cloud_corner_from_map.points[m_point_search_Idx[j]].y,
                                                    laser_cloud_corner_from_map.points[m_point_search_Idx[j]].z);
                                center = center + tmp;
                                nearCorners.push_back(tmp);
                            }

                            center = center / ((float)line_search_num);

                            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();

                            for (int j = 0; j < line_search_num; j++)
                            {
                                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                            }

                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                            // note Eigen library sort eigenvalues in increasing order
                            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                            {
                                line_is_avail = true;
                            }
                            else
                            {
                                line_is_avail = false;
                            }
                        }

                        // 对这5个points组成的集合计算协方差，特征值分解，最大特征值是否是次大特征的3倍以上，
                        // 若不是，忽略掉该点，拒绝points num++；
                        // 若是，计算残差

                        Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                        if (line_is_avail) // 1
                        {
                            if (ICP_LINE) // 1, //loop clousre(scene_alignment.hpp)时：0, 不用corners计算loop closure
                            {
                                ceres::CostFunction *cost_function;
                                auto pt_1 = pcl_pt_to_eigend(laser_cloud_corner_from_map.points[m_point_search_Idx[0]]); // j point
                                auto pt_2 = pcl_pt_to_eigend(laser_cloud_corner_from_map.points[m_point_search_Idx[1]]); // l point
                                if ((pt_1 - pt_2).norm() < 0.0001)
                                    continue;

                                if (m_if_motion_deblur) // 线性插值
                                {
                                    cost_function = ceres_icp_point2line_mb<double>::Create(curr_point,
                                                                                            pt_1,
                                                                                            pt_2,
                                                                                            refine_blur(pointOri.intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp) * 1.0,
                                                                                            Eigen::Matrix<double, 4, 1>(m_q_w_last.w(), m_q_w_last.x(), m_q_w_last.y(), m_q_w_last.z()),
                                                                                            m_t_w_last); //pointOri.intensity );
                                }
                                else
                                { // 分段
                                    cost_function = ceres_icp_point2line<double>::Create(curr_point,
                                                                                         pt_1,
                                                                                         pt_2,
                                                                                         Eigen::Matrix<double, 4, 1>(m_q_w_last.w(), m_q_w_last.x(), m_q_w_last.y(), m_q_w_last.z()),
                                                                                         m_t_w_last);
                                }

                                // 直线到点的距离向量，由垂足指向点
                                block_id = problem.AddResidualBlock(cost_function, loss_function, m_para_buffer_incremental, m_para_buffer_incremental + 4);
                                residual_block_ids.push_back(block_id);
                                corner_avail_num++;
                            }
                        }
                        else
                        { // 忽略掉该point
                            corner_rejection_num++;
                        }
                    }
                } // end curr帧corners

                for (int i = 0; i < laser_surface_pt_num; i++)
                {
                    if (laser_surface_pt_num > 2 * m_maximum_allow_residual_block)
                    {
                        if (m_rand_float.rand_uniform() * laser_surface_pt_num > 2 * m_maximum_allow_residual_block)
                        {
                            continue;
                        }
                    }

                    pointOri = laserCloudSurfStack->points[i]; // <2>history finished keyframe plannes(points in map)
                    int planeValid = true;
                    pointAssociateToMap(&pointOri, &pointSel, refine_blur(pointOri.intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp), if_undistore_in_matching);
                    m_kdtree_surf_from_map.nearestKSearch(pointSel, plane_search_num, m_point_search_Idx, m_point_search_sq_dis);

                    if (m_point_search_sq_dis[plane_search_num - 1] < m_maximum_dis_plane_for_match) // 50m
                    {
                        std::vector<Eigen::Vector3d> nearCorners;
                        Eigen::Vector3d center(0, 0, 0);
                        if (IF_PLANE_FEATURE_CHECK)
                        {
                            for (int j = 0; j < plane_search_num; j++)
                            {
                                Eigen::Vector3d tmp(laser_cloud_corner_from_map.points[m_point_search_Idx[j]].x,
                                                    laser_cloud_corner_from_map.points[m_point_search_Idx[j]].y,
                                                    laser_cloud_corner_from_map.points[m_point_search_Idx[j]].z);
                                center = center + tmp;
                                nearCorners.push_back(tmp);
                            }

                            center = center / (float)(plane_search_num);

                            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();

                            for (int j = 0; j < plane_search_num; j++)
                            {
                                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                            }

                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                            if ((saes.eigenvalues()[2] > 3 * saes.eigenvalues()[0]) &&
                                (saes.eigenvalues()[2] < 10 * saes.eigenvalues()[1]))
                            {
                                planeValid = true;
                            }
                            else
                            {
                                planeValid = false;
                            }
                        }

                        Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);

                        if (planeValid)
                        {
                            if (ICP_PLANE)
                            {
                                ceres::CostFunction *cost_function;

                                if (m_if_motion_deblur) // 线性插值； loop closure 时为0
                                {
                                    cost_function = ceres_icp_point2plane_mb<double>::Create(
                                        curr_point,
                                        pcl_pt_to_eigend(laser_cloud_surf_from_map.points[m_point_search_Idx[0]]),
                                        pcl_pt_to_eigend(laser_cloud_surf_from_map.points[m_point_search_Idx[plane_search_num / 2]]),
                                        pcl_pt_to_eigend(laser_cloud_surf_from_map.points[m_point_search_Idx[plane_search_num - 1]]),
                                        refine_blur(pointOri.intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp) * 1.0,
                                        Eigen::Matrix<double, 4, 1>(m_q_w_last.w(), m_q_w_last.x(), m_q_w_last.y(), m_q_w_last.z()),
                                        m_t_w_last);
                                }
                                else
                                {
                                    cost_function = ceres_icp_point2plane<double>::Create(
                                        curr_point,
                                        pcl_pt_to_eigend(laser_cloud_surf_from_map.points[m_point_search_Idx[0]]),
                                        pcl_pt_to_eigend(laser_cloud_surf_from_map.points[m_point_search_Idx[plane_search_num / 2]]),
                                        pcl_pt_to_eigend(laser_cloud_surf_from_map.points[m_point_search_Idx[plane_search_num - 1]]),
                                        Eigen::Matrix<double, 4, 1>(m_q_w_last.w(), m_q_w_last.x(), m_q_w_last.y(), m_q_w_last.z()),
                                        m_t_w_last); //pointOri.intensity );
                                }

                                // 残差：平面到点的向量，由垂足到点
                                block_id = problem.AddResidualBlock(cost_function, loss_function, m_para_buffer_incremental, m_para_buffer_incremental + 4);
                                residual_block_ids.push_back(block_id);
                            }
                            surf_avail_num++;
                        }
                        else
                        {
                            surface_rejecetion_num++;
                        }
                    }
                }

                std::vector<ceres::ResidualBlockId> residual_block_ids_temp;
                residual_block_ids_temp.reserve(residual_block_ids.size());

                // Drop some of the residual to guaruntee the real time performance.
                if (residual_block_ids.size() > (size_t)m_maximum_allow_residual_block)
                {
                    residual_block_ids_temp.clear();

                    float threshold_to_reserve = (float)m_maximum_allow_residual_block / (float)residual_block_ids.size();
                    float *probability_to_drop = m_rand_float.rand_array_uniform(0, 1.0, residual_block_ids.size());
                    screen_out << "Number of residual blocks too Large, drop them to " << m_maximum_allow_residual_block << endl;
                    for (size_t i = 0; i < residual_block_ids.size(); i++)
                    {
                        if (probability_to_drop[i] > threshold_to_reserve)
                        {
                            problem.RemoveResidualBlock(residual_block_ids[i]);
                        }
                        else
                        {
                            residual_block_ids_temp.push_back(residual_block_ids[i]);
                        }
                    }

                    residual_block_ids = residual_block_ids_temp;
                    delete probability_to_drop;
                }

                ceres::Solver::Options options;

                // If the number of residual block too Large, randomly drop some of them to guarentee the real-time perfromance.
                // 先让ceres迭代1次，根据结果的残差项大小，大约删除掉20%的outliers
                for (size_t ii = 0; ii < 1; ii++)
                {
                    options.linear_solver_type = slover_type;
                    options.max_num_iterations = m_para_cere_max_iterations;
                    options.max_num_iterations = m_para_cere_prerun_times; // 先让ceres迭代2次，根据结果的残差项大小，大约删除掉20%的outliers
                    options.minimizer_progress_to_stdout = false;
                    options.check_gradients = false; // TODO true时, 总是 Gradient error detected. Terminating solver
                    options.gradient_check_relative_precision = 1e-10;

                    set_ceres_solver_bound(problem, m_para_buffer_incremental); // 设置参数 m_t_w_incre最小值，最大值[-2，2]
                    ceres::Solve(options, &problem, &summary);

                    residual_block_ids_temp.clear();
                    ceres::Problem::EvaluateOptions eval_options;
                    eval_options.residual_blocks = residual_block_ids;
                    double total_cost = 0.0;
                    std::vector<double> residuals; // 残差是：点到直线的距离向量，点到面的距离向量
                    problem.Evaluate(eval_options, &total_cost, &residuals, nullptr, nullptr);

                    double m_inliner_ratio_threshold = compute_inlier_residual_threshold(residuals, m_inlier_ratio); // 内点所占比例大致为80%
                    m_inlier_threshold = std::max(m_inliner_dis, m_inliner_ratio_threshold);                         // 计算位姿时, max(0.02,  ); 闭环匹配时: max(0.2, )
                    for (unsigned int i = 0; i < residual_block_ids.size(); i++)
                    {
                        if ((fabs(residuals[3 * i + 0]) + fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2])) > m_inlier_threshold) // std::min( 1.0, 10 * avr_cost )
                        {
                            problem.RemoveResidualBlock(residual_block_ids[i]);
                        }
                        else
                        {
                            residual_block_ids_temp.push_back(residual_block_ids[i]);
                        }
                    }

                    residual_block_ids = residual_block_ids_temp;
                }

                options.linear_solver_type = slover_type;
                options.max_num_iterations = m_para_cere_max_iterations;
                options.minimizer_progress_to_stdout = false;
                options.check_gradients = false; // TODO true时, 总是 Gradient error detected. Terminating solver
                options.gradient_check_relative_precision = 1e-10;

                set_ceres_solver_bound(problem, m_para_buffer_incremental);
                ceres::Solve(options, &problem, &summary);
                if (m_if_motion_deblur) // 线性插值； loop closure时为0
                {
                    compute_interpolatation_rodrigue(q_w_incre, m_interpolatation_omega, m_interpolatation_theta, m_interpolatation_omega_hat);
                    m_interpolatation_omega_hat_sq2 = m_interpolatation_omega_hat * m_interpolatation_omega_hat;
                }

                m_t_w_curr = m_q_w_last * t_w_incre + m_t_w_last; // loop closure时，last=(I,0)
                m_q_w_curr = m_q_w_last * q_w_incre;              // loop closure时, curr = incre

                m_angular_diff = (float)m_q_w_curr.angularDistance(m_q_w_last) * 57.3; // delta角度
                m_t_diff = (m_t_w_curr - m_t_w_last).norm();                           // delata平移
                minimize_cost = summary.final_cost;

                if (q_last_optimize.angularDistance(q_w_incre) < 57.3 * m_minimum_icp_R_diff &&
                    (t_last_optimize - t_w_incre).norm() < m_minimum_icp_T_diff)
                {
                    screen_out << "----- Terminate, iteration times  = " << iterCount << "-----" << endl;
                    break;
                }
                else
                {
                    q_last_optimize = q_w_incre;
                    t_last_optimize = t_w_incre;
                }
            } // end icp 计算位姿

            screen_printf("===== corner factor num %d , surf factor num %d=====\n", corner_avail_num, surf_avail_num);
            if (laser_corner_pt_num != 0 && laser_surface_pt_num != 0)
            {
                m_logger_common->printf("Corner  total num %d |  use %d | rate = %d \% \r\n", laser_corner_pt_num, corner_avail_num, (corner_avail_num)*100 / laser_corner_pt_num);
                m_logger_common->printf("Surface total num %d |  use %d | rate = %d \% \r\n", laser_surface_pt_num, surf_avail_num, (surf_avail_num)*100 / laser_surface_pt_num);
            }
            
            *(m_logger_timer->get_ostream()) << m_timer->toc_string("Pose optimization") << std::endl;
            if (g_export_full_count < 5)
            {
                *(m_logger_common->get_ostream()) << summary.FullReport() << endl;
                g_export_full_count++;
            }
            else
            {
                *(m_logger_common->get_ostream()) << summary.BriefReport() << endl;
            }

            *(m_logger_common->get_ostream()) << "Last R:" << m_q_w_last.toRotationMatrix().eulerAngles(0, 1, 2).transpose() * 57.3 << " ,T = " << m_t_w_last.transpose() << endl;
            *(m_logger_common->get_ostream()) << "Curr R:" << m_q_w_curr.toRotationMatrix().eulerAngles(0, 1, 2).transpose() * 57.3 << " ,T = " << m_t_w_curr.transpose() << endl;
            *(m_logger_common->get_ostream()) << "Iteration time: " << iterCount << endl;

            m_logger_common->printf("Motion blur = %d | ", m_if_motion_deblur);
            m_logger_common->printf("Cost = %.5f| inlier_thr = %.2f |blk_size = %d | corner_num = %d | surf_num = %d | angle dis = %.2f | T dis = %.2f \r\n",
                                    minimize_cost, m_inlier_threshold, summary.num_residual_blocks, corner_avail_num, surf_avail_num, m_angular_diff, m_t_diff);

            m_inlier_threshold = m_inlier_threshold * summary.final_cost / summary.initial_cost; //
            // 计算位姿结束; 计算最终的匹配, used in loop closure(scene_alignment.hpp)

            // 计算位姿时:　　　　　delta_angle > 4度 　或 　最终的cost > 2.0, 　认为本次计算失败，当前帧的位姿置为上一帧的位姿
            // loop closure时:  没有通过函数的返回值是0 or 1 来判断是否对齐, 而是通过m_inlier_threshold来判断
            if (m_angular_diff > m_para_max_angular_rate || minimize_cost > m_max_final_cost)
            {
                *(m_logger_common->get_ostream()) << "**** Reject update **** " << endl;
                *(m_logger_common->get_ostream()) << summary.FullReport() << endl;
                for (int i = 0; i < 7; i++)
                {
                    m_para_buffer_RT[i] = m_para_buffer_RT_last[i]; // 其他地方没有使用这两个变量，没有用到
                }

                m_last_time_stamp = m_minimum_pt_time_stamp;
                m_q_w_curr = m_q_w_last;
                m_t_w_curr = m_t_w_last;

                return 0;
            }

            m_final_opt_summary = summary;
        }
        else
        {
            screen_out << "time Map corner and surf num are not enough" << std::endl;
        }

        return 1;
    }

    // called by scene_alignment.hpp(闭环)
    int find_out_incremental_transfrom(pcl::PointCloud<PointType>::Ptr in_laser_cloud_corner_from_map, // curr finished keyframe
                                       pcl::PointCloud<PointType>::Ptr in_laser_cloud_surf_from_map,
                                       pcl::PointCloud<PointType>::Ptr laserCloudCornerStack, // history finished keyframe
                                       pcl::PointCloud<PointType>::Ptr laserCloudSurfStack)
    {
        pcl::PointCloud<PointType> laser_cloud_corner_from_map = *in_laser_cloud_corner_from_map;
        pcl::PointCloud<PointType> laser_cloud_surf_from_map = *in_laser_cloud_surf_from_map;
        if (laser_cloud_corner_from_map.points.size() && laser_cloud_surf_from_map.points.size())
        {
            m_kdtree_corner_from_map.setInputCloud(laser_cloud_corner_from_map.makeShared());
            m_kdtree_surf_from_map.setInputCloud(laser_cloud_surf_from_map.makeShared());
        }
        else
        {
            return 1;
        }
        
        return find_out_incremental_transfrom(in_laser_cloud_corner_from_map, in_laser_cloud_surf_from_map, m_kdtree_corner_from_map, m_kdtree_surf_from_map, laserCloudCornerStack, laserCloudSurfStack);
    }

    /**
 * @param q_in:       输入的四元数
 * @param angle_axis: 旋转方向向量
 * @param angle_theta:旋转的角度
 * @param hat： 旋转方向向量对应的反对称矩阵 
 */
    void compute_interpolatation_rodrigue(const Eigen::Quaterniond &q_in, Eigen::Matrix<double, 3, 1> &angle_axis, double &angle_theta, Eigen::Matrix<double, 3, 3> &hat)
    {
        Eigen::AngleAxisd newAngleAxis(q_in);
        angle_axis = newAngleAxis.axis();
        angle_axis = angle_axis / angle_axis.norm();
        angle_theta = newAngleAxis.angle();
        hat.setZero();
        hat(0, 1) = -angle_axis(2);
        hat(1, 0) = angle_axis(2);
        hat(0, 2) = angle_axis(1);
        hat(2, 0) = -angle_axis(1);
        hat(1, 2) = -angle_axis(0);
        hat(2, 1) = angle_axis(0);
    };

    /**
 * @param pi: 当前帧下点 
 * @param po：在map下坐标
 * @param interpolate_s 
 * @param if_undistore : 是否进行去畸变
 * 把点转化到map下
 */
    // loop clousre(scene_alignment.hpp)时：interpolate_s = 1.0
    void pointAssociateToMap(PointType const *const pi, PointType *const po,
                             double interpolate_s = 1.0, int if_undistore = 0)
    {
        Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
        Eigen::Vector3d point_w;
        if (m_if_motion_deblur == 0 || if_undistore == 0 || interpolate_s == 1.0)
        {
            point_w = m_q_w_curr * point_curr + m_t_w_curr;
        }
        else
        {
            if (interpolate_s > 1.0 || interpolate_s < 0.0)
            {
                screen_printf("Input interpolate_s = %.5f\r\n", interpolate_s);
            }

            if (1) // Using rodrigues for fast compute.
            {
                //https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
                Eigen::Vector3d interpolate_T = m_t_w_incre * (interpolate_s * BLUR_SCALE);
                double interpolate_R_theta = m_interpolatation_theta * interpolate_s;
                Eigen::Matrix<double, 3, 3> interpolate_R_mat;

                interpolate_R_mat = Eigen::Matrix3d::Identity() + sin(interpolate_R_theta) * m_interpolatation_omega_hat + (1 - cos(interpolate_R_theta)) * m_interpolatation_omega_hat_sq2;
                point_w = m_q_w_last * (interpolate_R_mat * point_curr + interpolate_T) + m_t_w_last;
            }
            else
            {
                Eigen::Quaterniond interpolate_q = m_q_I.slerp(interpolate_s * BLUR_SCALE, m_q_w_incre);
                Eigen::Vector3d interpolate_T = m_t_w_incre * (interpolate_s * BLUR_SCALE);
                // 先转换到上一帧end下
                point_w = m_q_w_last * (interpolate_q * point_curr + interpolate_T) + m_t_w_last;
                // 再转换到map下
            }
        }

        po->x = point_w.x();
        po->y = point_w.y();
        po->z = point_w.z();
        po->intensity = pi->intensity;
    }

    void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
    {
        Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
        Eigen::Vector3d point_curr = m_q_w_curr.inverse() * (point_w - m_t_w_curr);
        po->x = point_curr.x();
        po->y = point_curr.y();
        po->z = point_curr.z();
        po->intensity = pi->intensity;
    }

    // called by laser_mapping.hpp
    unsigned int pointcloudAssociateToMap(pcl::PointCloud<PointType> const &pc_in, pcl::PointCloud<PointType> &pt_out,
                                          int if_undistore = 0)
    {
        unsigned int points_size = pc_in.points.size();
        pt_out.points.resize(points_size);

        for (unsigned int i = 0; i < points_size; i++)
        {
            pointAssociateToMap(&pc_in.points[i], &pt_out.points[i], refine_blur(pc_in.points[i].intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp), if_undistore);
        }

        return points_size;
    }

    unsigned int pointcloudAssociateTbeoMapped(pcl::PointCloud<PointType> const &pc_in, pcl::PointCloud<PointType> &pt_out)
    {
        unsigned int points_size = pc_in.points.size();
        pt_out.points.resize(points_size);

        for (unsigned int i = 0; i < points_size; i++)
        {
            pointAssociateTobeMapped(&pc_in.points[i], &pt_out.points[i]);
        }

        return points_size;
    }
};

#endif // POINT_CLOUD_REGISTRATION_HPP
