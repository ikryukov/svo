//
// Created by Alexey Klimov on 02.04.2022.
//

#pragma once

#include <Eigen/Core>
#include "map_point.h"


struct Camera {

    static struct cameraIntrinsics {
        double focal, cx, cy;
    } Intrinsics;

    void setRotation(const Eigen::Matrix3d& rotmat) {
        mEulerAngles = rotmat.eulerAngles(0, 1, 2); // todo is (0, 1, 2) right?
        mRotation = rotmat;
    }
    void setTranslation(const Eigen::Vector3d& tvec) {
        mTranslation = tvec;
    }

    Camera& rotate(const Eigen::Matrix3d& rotmat) {
        mRotation *= rotmat;
        mEulerAngles = mRotation.eulerAngles(0, 1, 2);
        return *this;
    }

    Camera& translate(const Eigen::Vector3d& tvec) {
        mTranslation += tvec;
        return *this;
    }

    Eigen::Vector3d& translation()               { return mTranslation; }
    Eigen::Matrix3d& rotationMat()               { return mRotation;    }
    Eigen::Vector3d& rotationEuler()             { return mEulerAngles; }
    const Eigen::Vector3d& translation() const   { return mTranslation; }
    const Eigen::Matrix3d& rotationMat() const   { return mRotation;    }
    const Eigen::Vector3d& rotationEuler() const { return mEulerAngles; }

private:

    Eigen::Vector3d mEulerAngles = Eigen::Vector3d::Identity();
    Eigen::Vector3d mTranslation = Eigen::Vector3d::Identity();
    Eigen::Matrix3d mRotation    = Eigen::Matrix3d::Identity();
};

struct KeyFrame {

    explicit KeyFrame(size_t start)
        : startID(start)
        , endID(start)
    {}

    ~KeyFrame() {
        for (auto* mp : mMapPoints) delete mp;
        for (auto* cam : mCameraPoses) delete cam;
    }

    Camera* getCamera(size_t id) {
        if (startID <= id && endID >= id)
            return mCameraPoses[id - startID];
        return nullptr;
    }

    MapPoint* getPoint(size_t mpID) {
        for (auto* mp : mMapPoints)
            if (mp->ID == mpID) return mp;
        return nullptr;
    }

    std::vector<MapPoint*> mMapPoints;
    std::vector<Camera*> mCameraPoses;
    size_t startID, endID;
};

