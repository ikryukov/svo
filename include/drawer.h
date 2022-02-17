//
// Created by Alexey Klimov on 04.02.2022.
//

#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

#include <Eigen/Geometry>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>


class MapPoint;


class Drawer {
public:
    explicit Drawer();
    void addCurrentPose(const cv::Matx<double, 3, 3>& rotation, const cv::Matx<double, 3, 1>& pose);
    void addMapPoints(const std::vector<MapPoint>& mapPoints);
    ~Drawer();

private:
    __forceinline void drawMapPoints();
    __forceinline void drawTrajectory();

    static void run(Drawer*);

    std::atomic<bool> mIsFinish = false;
    std::mutex mDrawerMutex;
    std::thread mThread;
    std::vector<float> mMapPoints;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
};
