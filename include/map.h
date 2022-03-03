//
// Created by Alexey Klimov on 13.02.2022.
//

#pragma once

#include <atomic>
#include <shared_mutex>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/types.hpp>

#include "drawer.h"
#include "map_point.h"


struct KeyFrame {
    ~KeyFrame() {
        for (auto* mp : mMapPoints)
            delete mp;
    }

    std::vector<MapPoint*> mMapPoints;
    std::vector<Eigen::Isometry3d> mPoses;
};


class Map {
    friend class Drawer;

public:

    Map();

    ~Map();

    void addPose(const cv::Matx<double, 3, 3>& rotation, const cv::Matx<double, 3, 1>& translation);

    void endKeyFrame();

    MapPoint* getPoint(size_t id);

    MapPoint* createMapPoint();
    MapPoint* createMapPoint(const Eigen::Vector3f& position, const std::vector<Observation>& observations = {});

    [[nodiscard]] size_t mapPointsSize() const;

private:

    static void run(Map* map);

    size_t mLastMapPointId = 0;
    std::vector<KeyFrame*> mKeyFrames;
    KeyFrame* mCurrentKeyFrame;

    mutable std::shared_mutex mCurrentKFMutex;
    mutable std::shared_mutex mMapMutex;
    std::thread mThread;
    std::atomic<bool> mIsFinish = false;

    Drawer mDrawer;
};
