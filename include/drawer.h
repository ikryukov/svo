//
// Created by Alexey Klimov on 04.02.2022.
//

#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

#include <opencv2/core/mat.hpp>


class MapPoint;


class Drawer {
public:
    explicit Drawer();
    void addCurrentPose(const cv::Mat& pose, const cv::Vec3f& r);
    void addMapPoints(const std::vector<MapPoint>& mapPoints);
    ~Drawer();

private:
    __forceinline static void drawCubeAt(const cv::Point3f& center, float edgeLength);
    __forceinline void drawMapPoints();
    __forceinline void drawTrajectory();
    __forceinline void drawCurrentPose();
    __forceinline void drawAllPoses();

    static void run(Drawer*);

    std::atomic<bool> mIsFinish = false;
    std::mutex mMutex;
    std::thread mThread;
    std::vector<cv::Point3f> mFeatures;
    cv::Mat pose;
    std::vector<float> mMapPoints;
    std::vector<double> mTrajectory;
    std::vector<double> mDirection;
    cv::Vec3f rotation_euler;
};
