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


struct KeyFrame;
class MapPoint;
class Camera;

class Map {
    friend class Drawer;

public:

    Map();
    Map(std::vector<Eigen::Matrix3d> rotations, std::vector<Eigen::Vector3d> translations);

    ~Map();

    void addPose(const cv::Matx<double, 3, 3>& rotation, const cv::Matx<double, 3, 1>& translation);

    void endKeyFrame();

    MapPoint* getPoint(size_t id) const;
    Camera* getCamera(size_t id) const;

    MapPoint* createMapPoint();
    MapPoint* createMapPoint(const Eigen::Vector3d& position);

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
    std::vector<Eigen::Matrix3d> mGTRotations;
    std::vector<Eigen::Vector3d> mGTTranslations;
};
