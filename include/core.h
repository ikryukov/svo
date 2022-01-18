//
// Created by Alexey on 17.01.2022.
//

#ifndef CORE_H_
#define CORE_H_

#include <opencv2/core/mat.hpp>

#include "bucket.h"
#include "map_point.h"


void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0,
                                 std::vector<cv::Point2f>& points1,
                                 std::vector<cv::Point2f>& points2,
                                 std::vector<cv::Point2f>& points3,
                                 std::vector<cv::Point2f>& points0_return,
                                 std::vector<uchar>& status0,
                                 std::vector<uchar>& status1,
                                 std::vector<uchar>& status2,
                                 std::vector<uchar>& status3,
                                 std::vector<int>& ages);


void circularMatching(cv::Mat& img_l_0,
                      cv::Mat& img_r_0,
                      cv::Mat& img_l_1,
                      cv::Mat& img_r_1,
                      std::vector<cv::Point2f>& points_l_0,
                      std::vector<cv::Point2f>& points_r_0,
                      std::vector<cv::Point2f>& points_l_1,
                      std::vector<cv::Point2f>& points_r_1,
                      std::vector<cv::Point2f>& points_l_0_return,
                      FeatureSet& current_features);


void checkValidMatch(const std::vector<cv::Point2f>& points,
                     const std::vector<cv::Point2f>& points_return,
                     std::vector<bool>& status,
                     float threshold);


void removeInvalidPoints(std::vector<cv::Point2f>& points, const std::vector<bool>& status);


void matchingFeatures(cv::Mat& imageLeft_t0,
                      cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1,
                      cv::Mat& imageRight_t1,
                      FeatureSet& currentVOFeatures,
                      std::vector<MapPoint> MapPoints,
                      std::vector<cv::Point2f>& pointsLeft_t0,
                      std::vector<cv::Point2f>& pointsRight_t0,
                      std::vector<cv::Point2f>& pointsLeft_t1,
                      std::vector<cv::Point2f>& pointsRight_t1);


void trackingFrame2Frame(const cv::Mat& projMatrl,
                         const cv::Mat& projMatrr,
                         std::vector<cv::Point2f>& pointsLeft_t0,
                         std::vector<cv::Point2f>& pointsLeft_t1,
                         cv::Mat& points3D_t0,
                         cv::Mat& rotation,
                         cv::Mat& translation);


void distinguishNewPoints(std::vector<cv::Point2f>& newPoints,
                          std::vector<bool>& valid,
                          std::vector<MapPoint>& mapPoints,
                          int frameId_t0,
                          cv::Mat& points3DFrame_t0,
                          cv::Mat& points3DFrame_t1,
                          cv::Mat& points3DWorld,
                          std::vector<cv::Point2f>& currentPointsLeft_t0,
                          std::vector<cv::Point2f>& currentPointsLeft_t1,
                          std::vector<FeaturePoint>& currentFeaturePointsLeft,
                          std::vector<FeaturePoint>& oldFeaturePointsLeft);


void integrateOdometryStereo(int frame_i,
                             cv::Mat& rigid_body_transformation,
                             cv::Mat& frame_pose,
                             const cv::Mat& rotation,
                             const cv::Mat& translation_stereo);

#endif // CORE_H_
