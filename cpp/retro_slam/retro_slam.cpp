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
#include "retro_slam.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sophus/se3.hpp>
#include <span>
#include <utility>
#include <vector>

#include "RetroSlamConfig.h"
#include "submap_graph.hpp"

namespace {
inline Vector3dVector trajectory_in_global_frame(const std::vector<Eigen::Matrix4d> &local_traj,
                                                 const Eigen::Matrix4d &keypose) {
    Vector3dVector traj(local_traj.size());
    const Eigen::Matrix3d R = keypose.block<3, 3>(0, 0);
    const Eigen::Vector3d t = keypose.block<3, 1>(0, 3);
    std::transform(local_traj.cbegin(), local_traj.cend(), traj.begin(),
                   [&](const Eigen::Matrix4d &pose) { return R * pose.block<3, 1>(0, 3) + t; });
    return traj;
}

std::vector<Eigen::Vector3d> stack_trajectories(
    const std::vector<std::vector<Eigen::Vector3d>> &trajectories) {
    const size_t num_points =
        std::transform_reduce(trajectories.cbegin(), trajectories.cend(), 0, std::plus<>(),
                              [](const std::vector<Eigen::Vector3d> &traj) { return traj.size(); });
    std::vector<Eigen::Vector3d> stacked_trajectories;
    stacked_trajectories.reserve(num_points);
    for (const auto &traj : trajectories) {
        stacked_trajectories.insert(stacked_trajectories.end(), traj.begin(), traj.end());
    }
    return stacked_trajectories;
}

inline double compute_hausdorff_distance(const std::vector<Eigen::Vector3d> &set1,
                                         const std::vector<Eigen::Vector3d> &set2,
                                         const double max_distance_threshold) {
    double max_distance_threshold_sq = max_distance_threshold * max_distance_threshold;
    double max_distance = 0.0;

    // Check distance from set1 to set2
    for (const auto &point1 : set1) {
        double min_distance = std::numeric_limits<double>::max();
        for (const auto &point2 : set2) {
            min_distance = std::min(min_distance, (point1 - point2).squaredNorm());
        }
        if (min_distance > max_distance_threshold_sq) {
            return std::numeric_limits<double>::max();
        }
        max_distance = std::max(max_distance, min_distance);
    }

    // Check distance from set2 to set1
    for (const auto &point2 : set2) {
        double min_distance = std::numeric_limits<double>::max();
        for (const auto &point1 : set1) {
            min_distance = std::min(min_distance, (point1 - point2).squaredNorm());
        }
        if (min_distance > max_distance_threshold_sq) {
            return std::numeric_limits<double>::max();
        }
        max_distance = std::max(max_distance, min_distance);
    }

    return std::sqrt(max_distance);
}
}  // namespace

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
}  // namespace Eigen

using retro_slam::Submap;
using Vector3dVector = Vector3dVector;

namespace retro_slam {
RetroSLAM::RetroSLAM(const RetroSlamConfig &config)
    : config_(config),
      hausdorff_distance_threshold_(config.odometry.max_range),
      odometry_pipeline_(config.odometry),
      submap_graph_(config.odometry),
      closer_(config.loop_closer),
      optimizer_(config.pose_graph_optimizer.max_iterations, config.pose_graph_optimizer.verbose),
      submap_voxel_grid_(config.local_mapper.submap_voxel_size),
      local_map_voxel_grid_(config.local_mapper.localmap_voxel_size) {
    optimizer_.AddVariable(0, Eigen::Matrix4d::Identity());
    optimizer_.FixVariable(0);
}

void RetroSLAM::ProcessScan(const Vector3dVector &scan_points,
                            const std::vector<double> &scan_timestamps) {
    const auto &[deskewed_frame, _] =
        odometry_pipeline_.RegisterFrame(scan_points, scan_timestamps);

    const Eigen::Matrix4d scan_to_odom_pose = odometry_pipeline_.pose().matrix();
    slam_state_.localmap_T_scan = slam_state_.odom_T_localmap.inverse() * scan_to_odom_pose;
    slam_state_.submap_T_scan = slam_state_.odom_T_submap.inverse() * scan_to_odom_pose;

    latest_scan_points_ = deskewed_frame;
    latest_scan_pose_ = submap_graph_.CurrentKeypose() * slam_state_.submap_T_scan.matrix();

    local_map_voxel_grid_.IntegrateFrame(deskewed_frame, slam_state_.localmap_T_scan);
    submap_voxel_grid_.IntegrateFrame(deskewed_frame, slam_state_.submap_T_scan);

    local_trajectory_.emplace_back(slam_state_.submap_T_scan.matrix());

    const bool split_local_map = slam_state_.localmap_T_scan.block<3, 1>(0, 3).norm() >
                                 config_.local_mapper.localmap_splitting_distance;
    const bool split_submap = slam_state_.submap_T_scan.block<3, 1>(0, 3).norm() >
                              config_.local_mapper.submap_splitting_distance;
    if (split_submap || split_local_map) {
        FinalizeSubmap({local_trajectory_, submap_voxel_grid_.Pointcloud()});
        ResetSubmap();
        slam_state_.odom_T_submap = scan_to_odom_pose;
    }
    if (split_local_map) {
        const auto &[pointcloud, voxel_means, voxel_normals] = FinalizeLocalMap();
        ComputeClosures(pointcloud, voxel_means, voxel_normals);
        ResetLocalMap();
        slam_state_.odom_T_localmap = scan_to_odom_pose;
        submap_ids_in_local_map_.clear();
    }
}

void RetroSLAM::FinalizeSubmap(const SubmapData &submap_data) {
    const auto &trajectory = submap_data.trajectory;
    const auto &pointcloud = submap_data.pointcloud;
    submap_ids_in_local_map_.emplace_back(submap_graph_.current_id);
    const Eigen::Matrix4d current_keypose = submap_graph_.CurrentKeypose();
    submap_graph_.FinalizeSubmap(trajectory, pointcloud);
    optimizer_.AddVariable(submap_graph_.current_id, submap_graph_.CurrentKeypose());
    optimizer_.AddFactor(submap_graph_.current_id, submap_graph_.current_id - 1, trajectory.back(),
                         Eigen::Matrix6d::Identity(), false);
}

void RetroSLAM::ResetSubmap() {
    const Vector3dVector retained_points = submap_voxel_grid_.PointsWithinRadiusFromLocation(
        slam_state_.submap_T_scan.block<3, 1>(0, 3), config_.odometry.max_range);
    submap_voxel_grid_.Clear();
    submap_voxel_grid_.IntegrateFrame(retained_points, slam_state_.submap_T_scan.inverse());
    local_trajectory_.erase(local_trajectory_.begin() + 1, local_trajectory_.end());
}

std::tuple<Vector3dVector, Vector3dVector, Vector3dVector> RetroSLAM::FinalizeLocalMap() {
    const Vector3dVector pointcloud = local_map_voxel_grid_.Pointcloud();
    auto [voxel_means, voxel_normals] = local_map_voxel_grid_.PerVoxelMeanAndNormal();
    return {std::move(pointcloud), std::move(voxel_means), std::move(voxel_normals)};
}

void RetroSLAM::ResetLocalMap() {
    const Vector3dVector retained_points = local_map_voxel_grid_.PointsWithinRadiusFromLocation(
        slam_state_.localmap_T_scan.block<3, 1>(0, 3), config_.odometry.max_range);
    local_map_voxel_grid_.Clear();
    local_map_voxel_grid_.IntegrateFrame(retained_points, slam_state_.localmap_T_scan.inverse());
}

bool RetroSLAM::ComputeClosures(const Vector3dVector &query_points,
                                const Vector3dVector &voxel_means,
                                const Vector3dVector &voxel_normals) {
    slam_state_.submaps_per_localmap.emplace_back(submap_ids_in_local_map_);
    const auto result = closer_.Compute(query_points, voxel_means, voxel_normals);
    if (result.success) {
        return ComputeLoopClosuresConstraints(result.ref_id, result.query_id,
                                              result.pose_constraint.matrix());
    }
    return false;
}

bool RetroSLAM::ComputeLoopClosuresConstraints(const int src_localmap_id,
                                               const int tgt_localmap_id,
                                               const Eigen::Matrix4d &pose_constraint) {
    // process source submaps
    const auto &src_submap_ids = slam_state_.submaps_per_localmap[src_localmap_id];
    std::span<Submap> src_submaps =
        std::span{submap_graph_.graph}.subspan(src_submap_ids.front(), src_submap_ids.size());

    const Eigen::Matrix4d src_first_keypose_inv = src_submaps.front().keypose.inverse();
    const Eigen::Matrix4d pose_constraint_times_first_keypose_inv =
        pose_constraint * src_first_keypose_inv;
    std::vector<Eigen::Matrix4d> src_keyposes(src_submaps.size());
    std::transform(src_submaps.begin(), src_submaps.end(), src_keyposes.begin(),
                   [&](const Submap &submap) {
                       return pose_constraint_times_first_keypose_inv * submap.keypose;
                   });

    std::vector<Vector3dVector> src_trajectories;
    src_trajectories.reserve(src_submaps.size());
    for (size_t i = 0; i < src_submaps.size(); ++i) {
        src_trajectories.emplace_back(
            trajectory_in_global_frame(src_submaps[i].traj, src_keyposes[i]));
    }

    // process target submaps
    const auto &tgt_submap_ids = slam_state_.submaps_per_localmap[tgt_localmap_id];
    const std::span<Submap> tgt_submaps =
        std::span{submap_graph_.graph}.subspan(tgt_submap_ids.front(), tgt_submap_ids.size());

    const Eigen::Matrix4d tgt_first_keypose_inv = tgt_submaps.front().keypose.inverse();
    std::vector<Eigen::Matrix4d> tgt_keyposes(tgt_submaps.size());
    std::transform(tgt_submaps.begin(), tgt_submaps.end(), tgt_keyposes.begin(),
                   [&](const Submap &submap) { return tgt_first_keypose_inv * submap.keypose; });

    std::vector<Vector3dVector> tgt_trajectories;
    tgt_trajectories.reserve(tgt_submaps.size());
    for (size_t i = 0; i < tgt_submaps.size(); ++i) {
        tgt_trajectories.emplace_back(
            trajectory_in_global_frame(tgt_submaps[i].traj, tgt_keyposes[i]));
    }

    const auto candidates = FindRegistrationCandidates(
        src_submaps, tgt_submaps, src_keyposes, tgt_keyposes, src_trajectories, tgt_trajectories);
    const auto results = RegisterCandidates(candidates);
    const bool closure_inserted = AddRegistrationFactors(results);
    if (closure_inserted) {
        OptimizePoseGraph();
        if (slam_state_.last_closure_query_id != -1) {
            ComputeGraphTraversalConstraints(tgt_localmap_id, src_localmap_id);
        }
        slam_state_.last_closure_query_id = tgt_localmap_id;
        slam_state_.last_closure_ref_id = src_localmap_id;
    }
    return closure_inserted;
}

void RetroSLAM::ComputeGraphTraversalConstraints(const int current_localmap_query_id,
                                                 const int current_localmap_ref_id) {
    bool closure_inserted = false;
    for (int tgt_localmap_id = slam_state_.last_closure_query_id + 1;
         tgt_localmap_id < current_localmap_query_id; ++tgt_localmap_id) {
        const auto &tgt_submap_ids = slam_state_.submaps_per_localmap[tgt_localmap_id];
        const std::span<Submap> tgt_submaps =
            std::span{submap_graph_.graph}.subspan(tgt_submap_ids.front(), tgt_submap_ids.size());

        std::vector<Eigen::Matrix4d> tgt_keyposes(tgt_submaps.size());
        std::transform(tgt_submaps.begin(), tgt_submaps.end(), tgt_keyposes.begin(),
                       [&](const Submap &submap) { return submap.keypose; });
        std::vector<Vector3dVector> tgt_trajectories;
        tgt_trajectories.reserve(tgt_submaps.size());
        for (const auto &submap : tgt_submaps) {
            tgt_trajectories.emplace_back(trajectory_in_global_frame(submap.traj, submap.keypose));
        }

        const auto stacked_target_trajectories = stack_trajectories(tgt_trajectories);
        for (int src_localmap_id =
                 std::min(current_localmap_ref_id, slam_state_.last_closure_ref_id) + 1;
             src_localmap_id < std::max(current_localmap_ref_id, slam_state_.last_closure_ref_id);
             ++src_localmap_id) {
            const auto &src_submap_ids = slam_state_.submaps_per_localmap[src_localmap_id];
            const std::span<Submap> src_submaps = std::span{submap_graph_.graph}.subspan(
                src_submap_ids.front(), src_submap_ids.size());

            // Local Maps too far away to be overlapping, skip registration
            if ((tgt_submaps[0].keypose.block<3, 1>(0, 3) -
                 src_submaps[0].keypose.block<3, 1>(0, 3))
                    .norm() > (2 * config_.odometry.max_range)) {
                continue;
            }

            std::vector<Eigen::Matrix4d> src_keyposes(src_submaps.size());
            std::transform(src_submaps.begin(), src_submaps.end(), src_keyposes.begin(),
                           [&](const Submap &submap) { return submap.keypose; });

            std::vector<Vector3dVector> src_trajectories;
            src_trajectories.reserve(src_submaps.size());
            for (const auto &submap : src_submaps) {
                src_trajectories.emplace_back(
                    trajectory_in_global_frame(submap.traj, submap.keypose));
            }

            const auto stacked_source_trajectories = stack_trajectories(src_trajectories);

            const double hausdorff_distance =
                compute_hausdorff_distance(stacked_source_trajectories, stacked_target_trajectories,
                                           config_.odometry.max_range);
            // Local Maps too far away to be overlapping, skip registration
            if (hausdorff_distance <= config_.odometry.max_range) {
                const auto candidates =
                    FindRegistrationCandidates(src_submaps, tgt_submaps, src_keyposes, tgt_keyposes,
                                               src_trajectories, tgt_trajectories);
                const auto results = RegisterCandidates(candidates);
                closure_inserted = AddRegistrationFactors(results);
            }
        }
    }
    if (closure_inserted) OptimizePoseGraph();
}

std::vector<RegistrationCandidate> RetroSLAM::FindRegistrationCandidates(
    const std::span<Submap> &src_submaps,
    const std::span<Submap> &tgt_submaps,
    const std::vector<Eigen::Matrix4d> &src_keyposes,
    const std::vector<Eigen::Matrix4d> &tgt_keyposes,
    const std::vector<Vector3dVector> &src_trajectories,
    const std::vector<Vector3dVector> &tgt_trajectories) {
    std::vector<double> hausdorff_distances(tgt_submaps.size());
    std::vector<RegistrationCandidate> candidates;
    candidates.reserve(src_submaps.size());
    for (size_t src_idx = 0; src_idx < src_submaps.size(); ++src_idx) {
        const Vector3dVector &src_traj = src_trajectories[src_idx];
        std::transform(tgt_trajectories.cbegin(), tgt_trajectories.cend(),
                       hausdorff_distances.begin(), [&](const auto &tgt_traj) {
                           return compute_hausdorff_distance(src_traj, tgt_traj,
                                                             hausdorff_distance_threshold_);
                       });
        const auto min_distance_it =
            std::min_element(hausdorff_distances.cbegin(), hausdorff_distances.cend());
        const auto tgt_idx = std::distance(hausdorff_distances.cbegin(), min_distance_it);
        if (*min_distance_it <= hausdorff_distance_threshold_) {
            const Submap &src_submap = src_submaps[src_idx];
            const Submap &tgt_submap = tgt_submaps[tgt_idx];
            const Eigen::Matrix4d initial_guess =
                tgt_keyposes[tgt_idx].inverse() * src_keyposes[src_idx];
            candidates.emplace_back(src_submap.id, tgt_submap.id, initial_guess);
        }
    }
    return candidates;
}

std::vector<RegistrationResult> RetroSLAM::RegisterCandidates(
    const std::vector<RegistrationCandidate> &candidates) {
    std::vector<RegistrationResult> results;
    results.reserve(candidates.size());
    for (const auto &candidate : candidates) {
        const auto &[pose, fitness_score] = submap_graph_.RegisterSubmaps(
            candidate.src_id, candidate.tgt_id, candidate.initial_guess);

        results.emplace_back(candidate.src_id, candidate.tgt_id, fitness_score, pose);
    }
    return results;
}

bool RetroSLAM::AddRegistrationFactors(const std::vector<RegistrationResult> &results) {
    bool valid_constraints = false;
    for (const auto &result : results) {
        if (result.fitness_score > config_.loop_closer.fitness_threshold) {
            valid_constraints = true;
            loop_closures.emplace_back(result.src_id, result.tgt_id);
            optimizer_.AddFactor(result.src_id, result.tgt_id, result.pose.matrix(),
                                 Eigen::Matrix6d::Identity(), false);
        }
    }
    return valid_constraints;
}

void RetroSLAM::OptimizePoseGraph() {
    optimizer_.Optimize();
    const auto &estimates = optimizer_.Estimates();
    std::for_each(submap_graph_.graph.begin(), submap_graph_.graph.end(),
                  [&](Submap &submap) { submap.keypose = estimates[submap.id]; });
}

void RetroSLAM::Exit() {
    if (local_trajectory_.size() > 1) {
        FinalizeSubmap({local_trajectory_, submap_voxel_grid_.Pointcloud()});
    }
    if (submap_ids_in_local_map_.size() > 0) {
        const auto &[pointcloud, voxel_means, voxel_normals] = FinalizeLocalMap();
        ComputeClosures(pointcloud, voxel_means, voxel_normals);
    }
    submap_graph_.EraseCurrentSubmap();
}

std::vector<Eigen::Matrix4d> RetroSLAM::Poses() const {
    std::vector<Eigen::Matrix4d> poses;
    poses.emplace_back(Eigen::Matrix4d::Identity());
    std::for_each(submap_graph_.graph.cbegin(), submap_graph_.graph.cend(),
                  [&](const Submap &submap) {
                      const auto &local_trajectory = submap.traj;
                      for (int i = 1; i < local_trajectory.size(); ++i) {
                          poses.emplace_back(submap.keypose * local_trajectory[i]);
                      }
                  });
    return poses;
}
}  // namespace retro_slam
