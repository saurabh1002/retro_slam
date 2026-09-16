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
#include <vector>

#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/pipeline/KissICP.hpp"
#include "voxel_map.hpp"

namespace {
constexpr double kernel_scale_factor = 3.0;
constexpr double voxel_map_max_range = 100.0;

constexpr double voxel_size = 0.5;
constexpr double icp_threshold = std::sqrt(kernel_scale_factor) * voxel_size;
}  // namespace

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
}

namespace retro_slam {
struct Result {
    Sophus::SE3d refined_pose;
    Eigen::Matrix6d hessian;
    double fitness_score;
};

struct Submap {
    Submap(const int id, const int scan_id, const Sophus::SE3d &keypose)
        : id(id), scan_id(scan_id), keypose(keypose) {
        traj.emplace_back(Sophus::SE3d());
    }

    void emplace_back(const Sophus::SE3d &pose) { traj.emplace_back(pose); }
    Sophus::SE3d endpose() const { return traj.size() > 0 ? keypose * traj.back() : keypose; }

    int id;
    int scan_id;
    Sophus::SE3d keypose;
    std::vector<Sophus::SE3d> traj;
    std::vector<Eigen::Vector3d> pcd;
};

struct SubmapGraph {
    SubmapGraph(const kiss_icp::pipeline::KISSConfig &odometry_config);

    Submap &operator[](const int id) { return graph.at(id); }

    std::vector<Eigen::Matrix4d> Keyposes() const;

    Submap &CurrentSubmap() { return graph.back(); }

    Sophus::SE3d CurrentKeypose() const { return graph.back().keypose; }

    void EraseCurrentSubmap() {
        graph.pop_back();
        current_id--;
    }

    void FinalizeSubmap(const std::vector<Sophus::SE3d> &local_trajectory,
                        const std::vector<Eigen::Vector3d> &pointcloud);

    Result RegisterSubmaps(const int source_id,
                           const int target_id,
                           const Sophus::SE3d &initial_guess) const;
    int current_id = 0;
    std::vector<Submap> graph;
    kiss_icp::pipeline::KISSConfig odometry_config_;
};
}  // namespace retro_slam
