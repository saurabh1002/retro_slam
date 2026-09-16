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
#include "submap_graph.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <sophus/se3.hpp>
#include <vector>

#include "kiss_icp/core/VoxelHashMap.hpp"
#include "kiss_icp/pipeline/KissICP.hpp"
#include "voxel_map.hpp"

namespace retro_slam {
SubmapGraph::SubmapGraph(const kiss_icp::pipeline::KISSConfig &odometry_config)
    : current_id(0), odometry_config_(odometry_config) {
    Submap first_submap(0, 0, Sophus::SE3d());
    first_submap.traj.clear();
    graph.emplace_back(std::move(first_submap));
}

std::vector<Eigen::Matrix4d> SubmapGraph::Keyposes() const {
    std::vector<Eigen::Matrix4d> poses(graph.size());
    std::transform(graph.cbegin(), graph.cend(), poses.begin(),
                   [](const auto &submap) { return submap.keypose.matrix(); });
    return poses;
}

void SubmapGraph::FinalizeSubmap(const std::vector<Sophus::SE3d> &local_trajectory,
                                 const std::vector<Eigen::Vector3d> &pointcloud) {
    Submap &submap = CurrentSubmap();
    submap.pcd = pointcloud;
    submap.traj = local_trajectory;

    current_id++;
    const int scan_id = submap.scan_id + static_cast<int>(submap.traj.size()) - 1;
    const Sophus::SE3d new_keypose = submap.endpose();
    graph.emplace_back(current_id, scan_id, new_keypose);
}

Result SubmapGraph::RegisterSubmaps(const int source_id,
                                    const int target_id,
                                    const Sophus::SE3d &initial_guess) const {
    kiss_icp::Registration registration_pipeline(odometry_config_.max_num_iterations,
                                                 odometry_config_.convergence_criterion,
                                                 odometry_config_.max_num_threads);
    const Vector3dVector &source_points = graph[source_id].pcd;
    const Vector3dVector &target_points = graph[target_id].pcd;
    kiss_icp::VoxelHashMap target_map(voxel_size, voxel_map_max_range);
    target_map.AddPoints(target_points);
    const Sophus::SE3d pose = registration_pipeline.AlignPointsToMap(
        source_points, target_map, initial_guess, icp_threshold, kernel_scale_factor);
    const Eigen::Matrix6d hessian =
        registration_pipeline.GetHessian(source_points, target_map, pose, icp_threshold);
    return {pose, hessian, registration_pipeline.fitness_};
}
}  // namespace retro_slam
