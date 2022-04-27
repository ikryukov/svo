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

void Map::addCamera(size_t frameID, const cv::Matx33d& rotmat, const cv::Matx31d& tvec) {
    Eigen::Matrix3d rotationEigen;
    cv::cv2eigen(rotmat, rotationEigen);
    Eigen::Vector3d translationEigen { tvec(0), tvec(1), tvec(2) };

    // Pick last known camera pose
    Eigen::Isometry3d lastPose = mAllCameras.empty() ? Eigen::Isometry3d::Identity()
                                                     : mAllCameras.back()->mCameraPose;

    // Transform to new position
    lastPose.rotate(rotationEigen).translate(translationEigen);

    Camera* newCamera;
    if (mCurrentKeyFrame->ID == frameID) {
        mCurrentKeyFrame->mCameraPose = lastPose;
        {
            std::unique_lock lock(mMapMutex);
            mKeyFrames.push_back(mCurrentKeyFrame);
        }
        newCamera = mCurrentKeyFrame;
    } else {
        newCamera = new Camera{ frameID, lastPose };
    }

    std::unique_lock lock(mMapMutex);
    mAllCameras.push_back(newCamera);
}

void Map::prepareKeyFrame(size_t frameID) {
    mCurrentKeyFrame = new KeyFrame{ frameID };
}

MapPoint* Map::getPoint(size_t id) {
    auto it = std::find_if(mCurrentKeyFrame->mMapPoints.begin(), mCurrentKeyFrame->mMapPoints.end(), [id](auto& mp) {
      return mp->ID == id;
    });

    if (it == mCurrentKeyFrame->mMapPoints.end()) {
        std::shared_lock allLock(mMapMutex);
        for (auto rit = mKeyFrames.rbegin(); rit != mKeyFrames.rend(); ++rit) {
            auto* kf = *rit;
            it = std::find_if(kf->mMapPoints.begin(), kf->mMapPoints.end(), [id](auto* mp) {
              return mp->ID == id;
            });
            if (it != kf->mMapPoints.end()) break;
        }
    }

    return *it;
}

MapPoint* Map::createMapPoint(const Eigen::Vector3f& position, const std::vector<Observation>& observations) {
    mCurrentKeyFrame->mMapPoints.push_back(new MapPoint(mLastMapPointId++, position, observations));
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

    return res + mCurrentKeyFrame->mMapPoints.size();
}

void Map::run(Map* map) {
    while (!map->mIsFinish) {
        // TODO optimization part here
    }
}
