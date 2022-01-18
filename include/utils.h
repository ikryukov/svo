//
// Created by Alexey on 17.01.2022.
//

#ifndef UTILS_H_
#define UTILS_H_

#include <opencv2/core/mat.hpp>

bool isRotationMatrix(cv::Mat& R);

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat& R);

void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose, std::vector<cv::Mat>& pose_matrix_gt, float fps, bool show_gt);

void displayTracking(cv::Mat& imageLeft_t1, std::vector<cv::Point2f>& pointsLeft_t0, std::vector<cv::Point2f>& pointsLeft_t1);

#endif // UTILS_H_
