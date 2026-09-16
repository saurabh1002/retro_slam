# MIT License
#
# Copyright (c) 2026 Saurabh Gupta.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np

from retro_slam.config import RetroSlamConfig
from retro_slam.pybind import retro_slam_pybind


class RetroSLAM:
    def __init__(self, config: RetroSlamConfig):
        self.config = config
        self.retro_slam = retro_slam_pybind._RetroSLAM(self.config.model_dump())

    def process_scan(self, frame: np.ndarray, timestamps: np.ndarray):
        self.retro_slam._process_scan(retro_slam_pybind._Vector3dVector(frame), timestamps.ravel())

    def write_pose_graph(self, filename: str):
        self.retro_slam._write_pose_graph(filename)

    def exit(self):
        self.retro_slam._exit()

    @property
    def poses(self) -> np.ndarray:
        return np.asarray(self.retro_slam._poses())

    @property
    def keyposes(self) -> np.ndarray:
        return np.asarray(self.retro_slam._keyposes())

    @property
    def loop_closures(self):
        return np.asarray(self.retro_slam.loop_closures_)
