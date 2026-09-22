// MIT License

// Copyright (c) 2026 Saurabh Gupta

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <span>
#include <string>
#include <vector>

#include "RetroSlamConfig.h"
#include "kiss_icp/pipeline/KissICP.hpp"
#include "loop_closer.hpp"
#include "pose_graph_optimizer.hpp"
#include "submap_graph.hpp"
#include "voxel_map.hpp"

namespace {
struct RegistrationCandidate {
    int src_id;
    int tgt_id;
    int src_scan_id;
    int tgt_scan_id;
    Eigen::Matrix4d initial_guess;

    RegistrationCandidate(const int src_id,
                          const int tgt_id,
                          const int src_scan_id,
                          const int tgt_scan_id,
                          const Eigen::Matrix4d &initial_guess)
        : src_id(src_id),
          tgt_id(tgt_id),
          src_scan_id(src_scan_id),
          tgt_scan_id(tgt_scan_id),
          initial_guess(initial_guess) {}
};

struct RegistrationResult {
    int src_id;
    int tgt_id;
    int src_scan_id;
    int tgt_scan_id;

    double fitness_score;
    Eigen::Matrix4d pose;
    Eigen::Matrix6d hessian;

    RegistrationResult(const int src_id,
                       const int tgt_id,
                       const int src_scan_id,
                       const int tgt_scan_id,
                       const double fitness_score,
                       const Eigen::Matrix4d &pose,
                       const Eigen::Matrix6d &hessian)
        : src_id(src_id),
          tgt_id(tgt_id),
          src_scan_id(src_scan_id),
          tgt_scan_id(tgt_scan_id),
          fitness_score(fitness_score),
          pose(pose),
          hessian(hessian) {}
};
}  // namespace

namespace retro_slam {
struct SlamState {
    Eigen::Matrix4d odom_T_localmap = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d localmap_T_scan = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d odom_T_submap = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d submap_T_scan = Eigen::Matrix4d::Identity();

    std::vector<std::vector<int>> submaps_per_localmap;
};

struct Closures {
    int ref_scan_id;
    int query_scan_id;
    Closures(const int ref_id, const int query_id) : ref_scan_id(ref_id), query_scan_id(query_id) {}
};

struct SubmapData {
    std::vector<Eigen::Matrix4d> trajectory;
    std::vector<Eigen::Matrix<double, 6, 6>> hessians;
    std::vector<Eigen::Vector3d> pointcloud;
};

struct VisualizationState {
    std::vector<Eigen::Matrix4d> global_trajectory;
    std::vector<Eigen::Matrix4d> local_trajectory;
    std::vector<Eigen::Matrix4d> keyposes;
    std::vector<Eigen::Vector3d> latest_scan_points;
    Eigen::Matrix4d latest_scan_pose = Eigen::Matrix4d::Identity();
    std::vector<Closures> loop_closures;
};

class RetroSLAM {
public:
    explicit RetroSLAM(const RetroSlamConfig &config);
    void ProcessScan(const std::vector<Eigen::Vector3d> &scan_points,
                     const std::vector<double> &scan_timestamps);
    void FinalizeSubmap(const SubmapData &submap_data);
    void ResetSubmap();

    std::tuple<std::vector<Eigen::Vector3d>,
               std::vector<Eigen::Vector3d>,
               std::vector<Eigen::Vector3d>>
    FinalizeLocalMap();
    void ResetLocalMap();
    bool ComputeClosures(const std::vector<Eigen::Vector3d> &query_points,
                         const std::vector<Eigen::Vector3d> &voxel_means,
                         const std::vector<Eigen::Vector3d> &voxel_normals);
    void WritePoseGraph(const std::string &filename) const { optimizer_.WriteGraph(filename); }
    void Exit();

    std::vector<Eigen::Matrix4d> Poses() const { return optimizer_.Estimates(); }
    std::vector<Eigen::Matrix4d> Keyposes() const { return submap_graph_.Keyposes(); }
    Eigen::Matrix4d CurrentKeypose() const { return submap_graph_.CurrentKeypose(); }
    const std::vector<retro_slam::Submap> &Submaps() const { return submap_graph_.graph; }
    const std::vector<Eigen::Vector3d> &LatestScanPoints() const { return latest_scan_points_; }
    const Eigen::Matrix4d &LatestScanPose() const { return latest_scan_pose_; }
    VisualizationState LatestVisualizationState() const {
        VisualizationState state;
        state.global_trajectory = optimizer_.Estimates();
        state.local_trajectory = local_trajectory_;
        state.keyposes = submap_graph_.Keyposes();
        state.latest_scan_points = latest_scan_points_;
        state.latest_scan_pose = latest_scan_pose_;
        state.loop_closures = loop_closures;
        return state;
    }

private:
    bool ComputeLoopClosuresConstraints(const int src_localmap_id,
                                        const int tgt_localmap_id,
                                        const Eigen::Matrix4d &pose_constraint);

    std::vector<RegistrationCandidate> FindRegistrationCandidates(
        const std::span<retro_slam::Submap> &src_submaps,
        const std::span<retro_slam::Submap> &tgt_submaps,
        const std::vector<Eigen::Matrix4d> &src_keyposes,
        const std::vector<Eigen::Matrix4d> &tgt_keyposes,
        const std::vector<Vector3dVector> &src_trajectories,
        const std::vector<Vector3dVector> &tgt_trajectories);

    std::vector<RegistrationResult> RegisterCandidates(
        const std::vector<RegistrationCandidate> &candidates);

    bool AddRegistrationFactors(const std::vector<RegistrationResult> &results);

    void OptimizePoseGraph();

private:
    RetroSlamConfig config_;

    SlamState slam_state_{};
    std::vector<Eigen::Matrix4d> local_trajectory_;
    std::vector<Eigen::Matrix<double, 6, 6>> hessians_;
    std::vector<int> submap_ids_in_local_map_;
    double hausdorff_distance_threshold_ = 0.0;

    kiss_icp::pipeline::KissICP odometry_pipeline_;
    SubmapGraph submap_graph_;
    LoopCloser closer_;
    PoseGraphOptimizer optimizer_;

    VoxelMap submap_voxel_grid_;
    VoxelMap local_map_voxel_grid_;
    std::vector<Eigen::Vector3d> latest_scan_points_;
    Eigen::Matrix4d latest_scan_pose_ = Eigen::Matrix4d::Identity();

public:
    std::vector<Closures> loop_closures;
};
}  // namespace retro_slam
