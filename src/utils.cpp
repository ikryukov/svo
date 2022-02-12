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

#include "utils.h"


bool isRotationMatrix(cv::Matx<double, 3, 3>& R) {
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    return cv::norm(I, shouldBeIdentity) < 1e-6;
}

cv::Vec3f rotationMatrixToEulerAngles(cv::Matx<double, 3, 3>& R) {
    assert(isRotationMatrix(R));

    float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R(2, 1), R(2, 2));
        y = atan2(-R(2, 0), sy);
        z = atan2(R(1, 0), R(0, 0));
    }
    else
    {
        x = atan2(-R(1, 2), R(1, 1));
        y = atan2(-R(2, 0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
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
    cv::waitKey(1);
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

void printSummary(std::pair<double, int> max, std::pair<double, int> min, double avg, double total, size_t maxRAM) {
    int seconds = (int)total;
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
