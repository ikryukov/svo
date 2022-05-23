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

#include "fbow/fbow.h"

#include "drawer.h"
#include "map_point.h"


struct Frame {
    size_t ID;
    bool isKeyFrame = false;
    Eigen::Isometry3d mCameraPose;
};


class Map {
    friend class Drawer;

public:

    Map();
    Map(std::vector<Eigen::Matrix3d> rotations, std::vector<Eigen::Vector3d> translations);

    ~Map();

    void updatePosition(size_t frameID, const cv::Matx33d& rotmat, const cv::Matx31d& tvec);

    void createKeyFrame(size_t frameID);

    void addDescriptor(cv::Mat& descriptor);

    MapPoint* getPoint(size_t id);

    MapPoint* createMapPoint(const Eigen::Vector3f& position, const std::vector<Observation>& observations = {});

    [[nodiscard]] size_t mapPointsSize() const;

private:

    static void run(Map* map);

    std::vector<Frame*> mAllFrames;
    std::vector<Frame*> mKeyFrames;
    std::vector<MapPoint*> mMapPoints;
    Frame* mCurrentKeyFrame = nullptr;   // Points to the last known keyframe

    mutable std::shared_mutex mMapMutex;
    std::thread mThread;
    std::atomic<bool> mIsFinish = false;

    Drawer mDrawer;

    fbow::Vocabulary vocab;
    std::vector<cv::Mat> mDescriptors;
    size_t mDescriptorsLoopChecked=1;

    std::vector<Eigen::Matrix3d> mGTRotations;
    std::vector<Eigen::Vector3d> mGTTranslations;
};
