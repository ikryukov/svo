//
// Created by Alexey Klimov on 27.02.2022.
//

#pragma once

#include <vector>
#include <shared_mutex>

#include <Eigen/Geometry>

#include "map_point.h"


struct KeyFrame {
    KeyFrame() = default;
    KeyFrame(const KeyFrame& other);
    KeyFrame(KeyFrame&& other) noexcept;
    KeyFrame& operator=(const KeyFrame& other);
    KeyFrame& operator=(KeyFrame&& other) noexcept;
    ~KeyFrame();

    std::shared_mutex mKeyFrameMutex;
    std::vector<MapPoint*> mMapPoints;
    std::vector<Eigen::Isometry3d> mPoses;
};
