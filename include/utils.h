//
// Created by Alexey on 17.01.2022.
//

#ifndef UTILS_H_
#define UTILS_H_

#include <opencv2/core/mat.hpp>

#include "windows.h"
#include "Psapi.h"


bool isRotationMatrix(cv::Mat& R);


// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat& R);


void display(int frame_id,
             const cv::Mat& trajectory,
             const cv::Mat& pose,
             const cv::Mat& translation,
             float fps,
             bool show_gt);


void displayTracking(cv::Mat& imageLeft_t1,
                     std::vector<cv::Point2f>& pointsLeft_t0,
                     std::vector<cv::Point2f>& pointsLeft_t1);


// returns total amount of physical memory (RAM)
size_t getTotalRAM();


// returns amount of physical memory (RAM) that current process using in MB
size_t getCurrentlyUsedRAM();


void printSummary(std::pair<double, int> max,
                  std::pair<double, int> min,
                  double avg,
                  time_t total,
                  size_t maxRAM);


#endif // UTILS_H_
