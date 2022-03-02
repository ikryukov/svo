//
// Created by Alexey Klimov on 13.02.2022.
//

#include "map.h"

#include <opencv2/core/eigen.hpp>


Map::Map()
    : mCurrentKeyFrame(new KeyFrame)
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

void Map::addPose(const cv::Matx<double, 3, 3>& rotation, const cv::Matx<double, 3, 1>& translation) {
    Eigen::Matrix<double, 3, 3> rotationEigen;
    cv::cv2eigen(rotation, rotationEigen);
    Eigen::Vector3d translationEigen { translation(0), translation(1), translation(2) };

    auto lastPose = Eigen::Isometry3d::Identity();
    if (std::shared_lock lock1(mCurrentKFMutex); !mCurrentKeyFrame->mPoses.empty()) {
        lastPose = mCurrentKeyFrame->mPoses.back();
    } else if (std::shared_lock lock2(mMapMutex); !mKeyFrames.empty()) {
        lastPose = mKeyFrames.back()->mPoses.back();
    }

    std::unique_lock lock(mCurrentKFMutex);
    mCurrentKeyFrame->mPoses.push_back(lastPose.rotate(rotationEigen).translate(translationEigen));
}

void Map::endKeyFrame() {
    if (mCurrentKeyFrame->mMapPoints.empty()) return;

    {
        std::unique_lock lock(mMapMutex);
        mKeyFrames.push_back(mCurrentKeyFrame);
    }

    mCurrentKeyFrame = new KeyFrame;
}

MapPoint* Map::getPoint(size_t id) {
    std::shared_lock currentLock(mCurrentKFMutex);
    auto it = std::find_if(mCurrentKeyFrame->mMapPoints.begin(), mCurrentKeyFrame->mMapPoints.end(), [id](auto& mp) {
      return mp->ID == id;
    });

    if (it == mCurrentKeyFrame->mMapPoints.end()) {
        std::shared_lock allLock(mMapMutex);
        for (auto* kf : mKeyFrames) {
            it = std::find_if(kf->mMapPoints.begin(), kf->mMapPoints.end(), [id](auto* mp) {
              return mp->ID == id;
            });
            if (it != kf->mMapPoints.end()) break;
        }
    }

    return *it;
}

MapPoint* Map::createMapPoint() {
    {
        std::unique_lock lock(mCurrentKFMutex);
        mCurrentKeyFrame->mMapPoints.push_back(new MapPoint(mLastMapPointId++));
    }
    return mCurrentKeyFrame->mMapPoints.back();
}

MapPoint* Map::createMapPoint(const Eigen::Vector3f& position, const std::vector<Observation>& observations) {
    {
        std::unique_lock lock(mCurrentKFMutex);
        mCurrentKeyFrame->mMapPoints.push_back(new MapPoint(mLastMapPointId++, position, observations));
    }
    return mCurrentKeyFrame->mMapPoints.back();
}

void Map::run(Map* map) {
    while (!map->mIsFinish) {
        // TODO optimization part here
    }
}
