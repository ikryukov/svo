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


struct Camera {
    size_t ID;

    explicit Camera(size_t frameID)
        : ID(frameID)
    {}

    Camera(size_t frameID, const Eigen::Isometry3d& pose)
        : ID(frameID)
        , mCameraPose(pose)
    {}

    Eigen::Isometry3d mCameraPose;
};

struct KeyFrame : public Camera {

    explicit KeyFrame(size_t frameID)
        : Camera(frameID)
    {}

    ~KeyFrame() {
        for (auto* mp : mMapPoints)
            delete mp;
    }

    std::vector<MapPoint*> mMapPoints;
};


class Map {
    friend class Drawer;

public:

    Map();
    Map(std::vector<Eigen::Matrix3d> rotations, std::vector<Eigen::Vector3d> translations);

    ~Map();

    void addCamera(size_t frameID, const cv::Matx33d& rotmat, const cv::Matx31d& tvec);

    void prepareKeyFrame(size_t frameID);

    MapPoint* getPoint(size_t id);

    MapPoint* createMapPoint(const Eigen::Vector3f& position, const std::vector<Observation>& observations = {});

    [[nodiscard]] size_t mapPointsSize() const;

private:

    static void run(Map* map);

    std::vector<Camera*> mAllCameras;
    std::vector<KeyFrame*> mKeyFrames;
    KeyFrame* mCurrentKeyFrame = nullptr;   // Points to the last known keyframe

    size_t mLastMapPointId = 0;

    mutable std::shared_mutex mMapMutex;
    std::thread mThread;
    std::atomic<bool> mIsFinish = false;

    Drawer mDrawer;
    std::vector<Eigen::Matrix3d> mGTRotations;
    std::vector<Eigen::Vector3d> mGTTranslations;
};
