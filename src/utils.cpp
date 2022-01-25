//
// Created by Alexey on 17.01.2022.
//

#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

#ifdef _WIN32
#include "windows.h"
#include "Psapi.h"
#endif

bool isRotationMatrix(cv::Mat& R) {
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    return cv::norm(I, shouldBeIdentity) < 1e-6;
}


cv::Vec3f rotationMatrixToEulerAngles(cv::Mat& R) {
    assert(isRotationMatrix(R));

    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
}


void display(int frame_id, const cv::Mat& trajectory, const cv::Mat& pose, const cv::Mat& gt_translation, float fps, bool show_gt) {
    // draw estimated trajectory
    int x = int(pose.at<double>(0)) + 300;
    int y = int(pose.at<double>(2)) + 100;
    circle(trajectory, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);

    // draw ground truth trajectory
    if (show_gt) {
         x = int(gt_translation.at<double>(0)) + 300;
         y = int(gt_translation.at<double>(2)) + 100;
         circle(trajectory, cv::Point(x, y) ,1, CV_RGB(0,255,0), 1);
    }
    // print info

    // rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    // sprintf(text, "FPS: %02f", fps);
    // putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    cv::imshow("Trajectory", trajectory);

    cv::waitKey(1);
}

void displayTracking(cv::Mat& imageLeft_t1, std::vector<cv::Point2f>& pointsLeft_t0, std::vector<cv::Point2f>& pointsLeft_t1) {
    // -----------------------------------------
    // Display feature racking
    // -----------------------------------------
    int radius = 2;
    cv::Mat vis;

    cv::cvtColor(imageLeft_t1, vis, cv::COLOR_GRAY2BGR, 3);

    for (int i = 0; i < pointsLeft_t0.size(); i++)
    {
        cv::circle(vis, cv::Point2f(pointsLeft_t0[i].x, pointsLeft_t0[i].y), radius, CV_RGB(0, 255, 0));
    }

    for (int i = 0; i < pointsLeft_t1.size(); i++)
    {
        cv::circle(vis, cv::Point2f(pointsLeft_t1[i].x, pointsLeft_t1[i].y), radius, CV_RGB(255, 0, 0));
    }

    for (int i = 0; i < pointsLeft_t1.size(); i++)
    {
        cv::line(vis, pointsLeft_t0[i], pointsLeft_t1[i], CV_RGB(0, 255, 0));
    }

    cv::imshow("vis ", vis);
}

size_t getTotalRAM() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return memInfo.ullTotalPhys / 1024 / 1024;
#else
    return 0;
#endif
}

size_t getCurrentlyUsedRAM() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    return pmc.WorkingSetSize / 1024 / 1024;
#else
    return 0;
#endif
}

void printSummary(std::pair<double, int> max, std::pair<double, int> min, double avg, time_t total, size_t maxRAM) {
    int seconds = static_cast<int>(static_cast<double>(total) / CLOCKS_PER_SEC);
    int minutes = seconds / 60;
    int hours = minutes / 60;

    minutes -= hours * 60;
    seconds -= hours * 60 * 60 + minutes * 60;

    std::printf("\n############################## Summary #################################\n"
        "-- max memory usage: %lluMbs\n"
        "-- max / min frame time: [%.3lf, #%d], [%.3lf, #%d]\n"
        "-- Average frame time: %lf\n"
        "-- Total time: %02d:%02d:%02d\n",

        maxRAM,
        max.first,
        max.second,
        min.first,
        min.second,
        avg,
        hours,
        minutes,
        seconds
    );
}
