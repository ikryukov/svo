//
// Created by Alexey Klimov on 13.02.2022.
//

#include "map.h"

#include <opencv2/core/eigen.hpp>


Map::Map()
    : mThread(run, this)
    , mDrawer(*this)
{}

Map::Map(std::vector<Eigen::Matrix3d> rotations, std::vector<Eigen::Vector3d> translations)
    : mGTRotations(std::move(rotations))
    , mGTTranslations(std::move(translations))
    , mThread(run, this)
    , mDrawer(*this)
{}

Map::~Map() {
    mIsFinish = true;
    mThread.join();

    for (auto* kf : mKeyFrames) {
        delete kf;
    }
}

void Map::updatePosition(size_t frameID, const cv::Matx33d& rotmat, const cv::Matx31d& tvec) {
    Eigen::Matrix3d rotationEigen;
    cv::cv2eigen(rotmat, rotationEigen);
    Eigen::Vector3d translationEigen { tvec(0), tvec(1), tvec(2) };

    // Pick last known camera pose
    Eigen::Isometry3d lastPose;
    {
        std::shared_lock lock(mMapMutex);
        lastPose = mAllFrames.empty() ? Eigen::Isometry3d::Identity()
                                      : mAllFrames.back()->mCameraPose;
    }

    // Transform to new position
    lastPose.rotate(rotationEigen).translate(translationEigen);

    Frame* newFrame;
    if (mCurrentKeyFrame->ID == frameID) {
        mCurrentKeyFrame->mCameraPose = lastPose;
        {
            std::unique_lock lock(mMapMutex);
            mKeyFrames.push_back(mCurrentKeyFrame);
        }
        newFrame = mCurrentKeyFrame;
    } else {
        newFrame = new Frame{ frameID, false, lastPose };
    }

    std::unique_lock lock(mMapMutex);
    mAllFrames.push_back(newFrame);
}

void Map::createKeyFrame(size_t frameID) {
    mCurrentKeyFrame = new Frame{ frameID, true };
}

MapPoint* Map::getPoint(size_t id) {
    std::shared_lock allLock(mMapMutex);
    return id < mMapPoints.size() ? mMapPoints[id] : nullptr;
}

MapPoint* Map::createMapPoint(const Eigen::Vector3f& position, const std::vector<Observation>& observations) {
    std::unique_lock lock(mMapMutex);
    mMapPoints.push_back(new MapPoint(mMapPoints.size(), position, observations));
    return mMapPoints.back();
}

size_t Map::mapPointsSize() const {
    std::shared_lock lock(mMapMutex);
    return mMapPoints.size();
}

void Map::run(Map* map) {
    while (!map->mIsFinish) {
        // TODO optimization part here
    }
}
