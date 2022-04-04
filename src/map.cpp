//
// Created by Alexey Klimov on 13.02.2022.
//

#include "map.h"
#include "utils.h"

#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>

#include "keyframe.h"
#include "solve_ba.h"


Map::Map()
    : mCurrentKeyFrame(new KeyFrame{0})
    , mThread(run, this)
    , mDrawer(*this)
{}

Map::Map(std::vector<Eigen::Matrix3d> rotations, std::vector<Eigen::Vector3d> translations)
    : mCurrentKeyFrame(new KeyFrame{0})
    , mGTRotations(std::move(rotations))
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
    delete mCurrentKeyFrame;
}

void Map::addPose(const cv::Matx<double, 3, 3>& rotation, const cv::Matx<double, 3, 1>& translation) {
    Eigen::Matrix3d rotationEigen;
    cv::cv2eigen(rotation, rotationEigen);
    Eigen::Vector3d translationEigen { translation(0), translation(1), translation(2) };

    Camera* lastPose;
    if (std::shared_lock lock1(mCurrentKFMutex); !mCurrentKeyFrame->mCameraPoses.empty()) {
        lastPose = mCurrentKeyFrame->mCameraPoses.back();
    } else if (std::shared_lock lock2(mMapMutex); !mKeyFrames.empty()) {
        lastPose = mKeyFrames.back()->mCameraPoses.back();
    } else {
        lastPose = new Camera;
    }

    lastPose->rotate(rotationEigen);
    lastPose->translate(translationEigen);

    std::unique_lock lock(mCurrentKFMutex);
//    ++mCurrentKeyFrame->endID;
    mCurrentKeyFrame->mCameraPoses.push_back(lastPose);
}

void Map::endKeyFrame() {
    if (mCurrentKeyFrame->mMapPoints.empty()) return;

    auto* kf = mCurrentKeyFrame;
    kf->endID = kf->startID + kf->mCameraPoses.size() - 1;

    mCurrentKeyFrame = new KeyFrame{ kf->endID + 1 };

    solveBACeres(*this, kf);

    std::unique_lock lock(mMapMutex);
    mKeyFrames.push_back(kf);
}

MapPoint* Map::getPoint(size_t id) const {
    {
        std::shared_lock currentLock(mCurrentKFMutex);
        if (auto point = mCurrentKeyFrame->getPoint(id))
            return point;
    }

    {
        std::shared_lock allLock(mMapMutex);
        for (auto* kf : mKeyFrames)
            if (auto point = kf->getPoint(id)) return point;
    }

    std::cout << "-! Cant find map point with id: " << id << std::endl;
    return nullptr;
}

Camera* Map::getCamera(size_t id) const {
    {
        std::shared_lock currentLock(mCurrentKFMutex);
        if (auto cam = mCurrentKeyFrame->getCamera(id))
            return cam;
    }

    {
       std::shared_lock allLock(mMapMutex);
       for (auto* kf : mKeyFrames)
           if (kf->startID <= id && kf->endID >= id)
               return kf->mCameraPoses[id - kf->startID];
    }

    std::cout << "-! Cant find camera from frame #" << id << std::endl;
    return nullptr;
}

MapPoint* Map::createMapPoint() {
    {
        std::unique_lock lock(mCurrentKFMutex);
        mCurrentKeyFrame->mMapPoints.push_back(new MapPoint(mLastMapPointId++));
    }
    return mCurrentKeyFrame->mMapPoints.back();
}

MapPoint* Map::createMapPoint(const Eigen::Vector3d& position) {
    {
        std::unique_lock lock(mCurrentKFMutex);
        mCurrentKeyFrame->mMapPoints.push_back(new MapPoint(mLastMapPointId++, position));
    }
    return mCurrentKeyFrame->mMapPoints.back();
}

size_t Map::mapPointsSize() const {
    size_t res = 0;
    {
        std::shared_lock lock(mMapMutex);
        for (auto& kf : mKeyFrames) {
            res += kf->mMapPoints.size();
        }
    }

    std::shared_lock lock1(mCurrentKFMutex);
    res += mCurrentKeyFrame->mMapPoints.size();
    return res;
}

void Map::run(Map* map) {
    while (!map->mIsFinish) {
        // TODO optimization part here
    }
}
