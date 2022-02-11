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
#include "pangolin/display/display.h"
#include "pangolin/gl/opengl_render_state.h"


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

inline double SIGN(double x) {
    return (x >= 0.0f) ? +1.0f : -1.0f;
}

inline double NORM(double a, double b, double c, double d) {
    return sqrt(a * a + b * b + c * c + d * d);
}

// quaternion = [w, x, y, z]'
Eigen::Isometry3d mRot2Quat(const cv::Mat& m) {
    double r11 = m.at<double>(0, 0);
    double r12 = m.at<double>(0, 1);
    double r13 = m.at<double>(0, 2);
    double r21 = m.at<double>(1, 0);
    double r22 = m.at<double>(1, 1);
    double r23 = m.at<double>(1, 2);
    double r31 = m.at<double>(2, 0);
    double r32 = m.at<double>(2, 1);
    double r33 = m.at<double>(2, 2);
    double q0 = (r11 + r22 + r33 + 1.0f) / 4.0f;
    double q1 = (r11 - r22 - r33 + 1.0f) / 4.0f;
    double q2 = (-r11 + r22 - r33 + 1.0f) / 4.0f;
    double q3 = (-r11 - r22 + r33 + 1.0f) / 4.0f;
    if (q0 < 0.0f) {
        q0 = 0.0f;
    }
    if (q1 < 0.0f) {
        q1 = 0.0f;
    }
    if (q2 < 0.0f) {
        q2 = 0.0f;
    }
    if (q3 < 0.0f) {
        q3 = 0.0f;
    }
    q0 = sqrt(q0);
    q1 = sqrt(q1);
    q2 = sqrt(q2);
    q3 = sqrt(q3);
    if (q0 >= q1 && q0 >= q2 && q0 >= q3) {
        q0 *= +1.0f;
        q1 *= SIGN(r32 - r23);
        q2 *= SIGN(r13 - r31);
        q3 *= SIGN(r21 - r12);
    }
    else if (q1 >= q0 && q1 >= q2 && q1 >= q3) {
        q0 *= SIGN(r32 - r23);
        q1 *= +1.0f;
        q2 *= SIGN(r21 + r12);
        q3 *= SIGN(r13 + r31);
    }
    else if (q2 >= q0 && q2 >= q1 && q2 >= q3) {
        q0 *= SIGN(r13 - r31);
        q1 *= SIGN(r21 + r12);
        q2 *= +1.0f;
        q3 *= SIGN(r32 + r23);
    }
    else if (q3 >= q0 && q3 >= q1 && q3 >= q2) {
        q0 *= SIGN(r21 - r12);
        q1 *= SIGN(r31 + r13);
        q2 *= SIGN(r32 + r23);
        q3 *= +1.0f;
    }
    else {
        printf("coding error\n");
    }
    double r = NORM(q0, q1, q2, q3);
    q0 /= r;
    q1 /= r;
    q2 /= r;
    q3 /= r;

    Eigen::Isometry3d res(Eigen::Quaternion(q0, q1, q2, q3));
    return res;
}

