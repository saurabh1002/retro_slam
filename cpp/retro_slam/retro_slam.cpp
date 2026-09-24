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
#include <optional>
#include <sophus/se3.hpp>
#include <span>
#include <utility>
#include <vector>

#include "RetroSlamConfig.h"
#include "submap_graph.hpp"

namespace {
std::vector<Eigen::Matrix4d> compute_localmap_trajectory(
    const std::span<retro_slam::Submap> &submaps) {
    std::vector<Eigen::Matrix4d> localmap_trajectory;
    const size_t num_poses =
        std::transform_reduce(submaps.begin(), submaps.end(), static_cast<size_t>(0), std::plus<>(),
                              [](const auto &submap) { return submap.traj.size(); });
    localmap_trajectory.reserve(num_poses);
    for (const auto &submap : submaps) {
        const auto &keypose = submap.keypose;
        for (const auto &pose : submap.traj) {
            localmap_trajectory.emplace_back(keypose * pose);
        }
    }
    return localmap_trajectory;
}

inline double compute_hausdorff_distance(const std::vector<Eigen::Matrix4d> &source_traj,
                                         const std::vector<Eigen::Matrix4d> &target_traj,
                                         const double max_distance_threshold) {
    double max_distance_threshold_sq = max_distance_threshold * max_distance_threshold;
    double max_distance = 0.0;

    // Check distance from source to target
    for (const auto &pose_src : source_traj) {
        double min_distance = std::numeric_limits<double>::max();
        for (const auto &pose_tgt : target_traj) {
            min_distance =
                std::min(min_distance,
                         (pose_src.block<3, 1>(0, 3) - pose_tgt.block<3, 1>(0, 3)).squaredNorm());
        }
        if (min_distance > max_distance_threshold_sq) {
            return std::numeric_limits<double>::max();
        }
        max_distance = std::max(max_distance, min_distance);
    }

    // Check distance from target to source
    for (const auto &pose_tgt : target_traj) {
        double min_distance = std::numeric_limits<double>::max();
        for (const auto &pose_src : source_traj) {
            min_distance =
                std::min(min_distance,
                         (pose_src.block<3, 1>(0, 3) - pose_tgt.block<3, 1>(0, 3)).squaredNorm());
        }
        if (min_distance > max_distance_threshold_sq) {
            return std::numeric_limits<double>::max();
        }
        max_distance = std::max(max_distance, min_distance);
    }

    return std::sqrt(max_distance);
}

inline double compute_hausdorff_distance(const std::vector<Eigen::Matrix4d> &source_traj,
                                         const std::vector<Eigen::Matrix4d> &target_traj,
                                         const Eigen::Matrix4d &relative_pose,
                                         const double max_distance_threshold) {
    std::vector<Eigen::Matrix4d> transformed_source_traj;
    transformed_source_traj.reserve(source_traj.size());
    for (const auto &pose : source_traj) {
        transformed_source_traj.emplace_back(relative_pose * pose);
    }
    return compute_hausdorff_distance(transformed_source_traj, target_traj, max_distance_threshold);
}
}  // namespace

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
}  // namespace Eigen

using retro_slam::Submap;
using Vector3dVector = std::vector<Eigen::Vector3d>;

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
    const Eigen::Matrix6d hessian = odometry_pipeline_.hessian();

    const Eigen::Matrix4d scan_to_odom_pose = odometry_pipeline_.pose().matrix();
    slam_state_.localmap_T_scan = slam_state_.odom_T_localmap.inverse() * scan_to_odom_pose;
    slam_state_.submap_T_scan = slam_state_.odom_T_submap.inverse() * scan_to_odom_pose;

    latest_scan_points_ = deskewed_frame;
    latest_scan_pose_ = submap_graph_.CurrentKeypose() * slam_state_.submap_T_scan.matrix();

    local_map_voxel_grid_.IntegrateFrame(deskewed_frame, slam_state_.localmap_T_scan);
    submap_voxel_grid_.IntegrateFrame(deskewed_frame, slam_state_.submap_T_scan);

    local_trajectory_.emplace_back(slam_state_.submap_T_scan.matrix());
    hessians_.emplace_back(hessian);

    const bool split_local_map = slam_state_.localmap_T_scan.block<3, 1>(0, 3).norm() >
                                 config_.local_mapper.localmap_splitting_distance;
    const bool split_submap = slam_state_.submap_T_scan.block<3, 1>(0, 3).norm() >
                              config_.local_mapper.submap_splitting_distance;
    if (split_submap || split_local_map) {
        FinalizeSubmap({local_trajectory_, hessians_, submap_voxel_grid_.Pointcloud()});
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
    const auto &hessians = submap_data.hessians;
    const auto &pointcloud = submap_data.pointcloud;
    submap_ids_in_local_map_.emplace_back(submap_graph_.current_id);
    const Eigen::Matrix4d current_keypose = submap_graph_.CurrentKeypose();
    for (int id = 1; id < static_cast<int>(trajectory.size()); ++id) {
        const int current_scan_id = submap_graph_.CurrentSubmap().scan_id + id;
        const Eigen::Matrix4d &previous_local_pose = trajectory[id - 1];
        const Eigen::Matrix4d &current_local_pose = trajectory[id];
        optimizer_.AddVariable(current_scan_id, current_keypose * current_local_pose);
        optimizer_.AddFactor(current_scan_id, current_scan_id - 1,
                             previous_local_pose.inverse() * current_local_pose, hessians[id],
                             false);
    }
    submap_graph_.FinalizeSubmap(trajectory, pointcloud);
}

void RetroSLAM::ResetSubmap() {
    const Vector3dVector retained_points = submap_voxel_grid_.PointsWithinRadiusFromLocation(
        slam_state_.submap_T_scan.block<3, 1>(0, 3), config_.odometry.max_range);
    submap_voxel_grid_.Clear();
    submap_voxel_grid_.IntegrateFrame(retained_points, slam_state_.submap_T_scan.inverse());
    local_trajectory_.erase(local_trajectory_.begin() + 1, local_trajectory_.end());
    hessians_.erase(hessians_.begin() + 1, hessians_.end());
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
    const auto &src_submap_ids = slam_state_.submaps_per_localmap[src_localmap_id];
    std::span<Submap> src_submaps =
        std::span{submap_graph_.graph}.subspan(src_submap_ids[0], src_submap_ids.size());

    const auto &tgt_submap_ids = slam_state_.submaps_per_localmap[tgt_localmap_id];
    const std::span<Submap> tgt_submaps =
        std::span{submap_graph_.graph}.subspan(tgt_submap_ids[0], tgt_submap_ids.size());

    const Eigen::Matrix4d rel_pose_constraint =
        tgt_submaps[0].keypose * pose_constraint * src_submaps[0].keypose.inverse();
    const auto candidates =
        FindRegistrationCandidates(src_submaps, tgt_submaps, rel_pose_constraint);

    const bool closure_inserted = AddRegistrationFactors(candidates);
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
            std::span{submap_graph_.graph}.subspan(tgt_submap_ids[0], tgt_submap_ids.size());

        const auto target_localmap_trajectory = compute_localmap_trajectory(tgt_submaps);
        for (int src_localmap_id =
                 std::min(current_localmap_ref_id, slam_state_.last_closure_ref_id) + 1;
             src_localmap_id < std::max(current_localmap_ref_id, slam_state_.last_closure_ref_id);
             ++src_localmap_id) {
            const auto &src_submap_ids = slam_state_.submaps_per_localmap[src_localmap_id];
            const std::span<Submap> src_submaps =
                std::span{submap_graph_.graph}.subspan(src_submap_ids[0], src_submap_ids.size());

            if ((tgt_submaps[0].keypose.block<3, 1>(0, 3) -
                 src_submaps[0].keypose.block<3, 1>(0, 3))
                    .squaredNorm() >
                (4 * config_.odometry.max_range * config_.odometry.max_range)) {
                continue;
            }

            const auto source_localmap_trajectory = compute_localmap_trajectory(src_submaps);
            const double hausdorff_distance = compute_hausdorff_distance(
                source_localmap_trajectory, target_localmap_trajectory, config_.odometry.max_range);
            if (hausdorff_distance <= config_.odometry.max_range) {
                const auto candidates = FindRegistrationCandidates(src_submaps, tgt_submaps,
                                                                   Eigen::Matrix4d::Identity());

                closure_inserted = AddRegistrationFactors(candidates) || closure_inserted;
            }
        }
    }
    if (closure_inserted) OptimizePoseGraph();
}

std::vector<RegistrationCandidate> RetroSLAM::FindRegistrationCandidates(
    const std::span<Submap> &src_submaps,
    const std::span<Submap> &tgt_submaps,
    const Eigen::Matrix4d &rel_pose_constraint) {
    std::vector<double> hausdorff_distances(tgt_submaps.size());
    std::vector<RegistrationCandidate> candidates;
    candidates.reserve(src_submaps.size());
    for (const Submap &src_submap : src_submaps) {
        double lowest_hausdorff_distance = hausdorff_distance_threshold_;
        std::optional<int> tgt_idx = std::nullopt;
        const auto &src_traj = src_submap.traj;
        const Eigen::Matrix4d src_keypose = src_submap.keypose;
        for (int idx = 0; idx < static_cast<int>(tgt_submaps.size()); ++idx) {
            const Submap &tgt_submap = tgt_submaps[idx];
            const Eigen::Matrix4d relative_pose =
                tgt_submap.keypose.inverse() * rel_pose_constraint * src_keypose;
            const double hausdorff_distance = compute_hausdorff_distance(
                src_traj, tgt_submap.traj, relative_pose, hausdorff_distance_threshold_);
            hausdorff_distances[idx] = hausdorff_distance;
            if (hausdorff_distance <= lowest_hausdorff_distance) {
                lowest_hausdorff_distance = hausdorff_distance;
                tgt_idx = idx;
            }
        }
        if (tgt_idx.has_value()) {
            const Submap &matched_tgt_submap = tgt_submaps[tgt_idx.value()];
            const Eigen::Matrix4d initial_guess =
                matched_tgt_submap.keypose.inverse() * rel_pose_constraint * src_submap.keypose;
            candidates.emplace_back(src_submap.id, matched_tgt_submap.id, src_submap.scan_id,
                                    matched_tgt_submap.scan_id, initial_guess);
        }
    }
    return candidates;
}

bool RetroSLAM::AddRegistrationFactors(const std::vector<RegistrationCandidate> &candidates) {
    bool closure_inserted = false;
    for (const auto &candidate : candidates) {
        const auto result = submap_graph_.RegisterSubmaps(candidate.src_id, candidate.tgt_id,
                                                          candidate.initial_guess);
        if (result.fitness_score > config_.loop_closer.fitness_threshold) {
            closure_inserted = true;
            loop_closures.emplace_back(candidate.src_id, candidate.tgt_id);
            optimizer_.AddFactor(candidate.src_scan_id, candidate.tgt_scan_id,
                                 result.refined_pose.matrix(), result.hessian, false);
        }
    }
    return closure_inserted;
}

void RetroSLAM::OptimizePoseGraph() {
    optimizer_.Optimize();
    const auto &estimates = optimizer_.Estimates();

    std::for_each(submap_graph_.graph.begin(), submap_graph_.graph.end() - 1, [&](Submap &submap) {
        const int submap_scan_id = submap.scan_id;
        submap.keypose = estimates[submap_scan_id];
        const Eigen::Matrix4d inv_keypose = submap.keypose.inverse();
        for (int i = 0; i < static_cast<int>(submap.traj.size()); ++i) {
            submap.traj[i] = inv_keypose * estimates[submap_scan_id + i];
        }
    });

    Submap &current_submap = submap_graph_.CurrentSubmap();
    current_submap.keypose = estimates[current_submap.scan_id];
}

void RetroSLAM::Exit() {
    if (local_trajectory_.size() > 1) {
        FinalizeSubmap({local_trajectory_, hessians_, submap_voxel_grid_.Pointcloud()});
    }
    if (submap_ids_in_local_map_.size() > 0) {
        const auto &[pointcloud, voxel_means, voxel_normals] = FinalizeLocalMap();
        ComputeClosures(pointcloud, voxel_means, voxel_normals);
    }
    submap_graph_.EraseCurrentSubmap();
}
}  // namespace retro_slam
