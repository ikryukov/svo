//
// Created by Alexey on 17.01.2022.
//

#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>

#include "core.h"
#include "utils.h"
#include "bucket.h"
#include "config_reader.h"
#include "map.h"


void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0,
                                 std::vector<cv::Point2f>& points1,
                                 std::vector<cv::Point2f>& points2,
                                 std::vector<cv::Point2f>& points3,
                                 std::vector<cv::Point2f>& points0_return,
                                 std::vector<uchar>& status0,
                                 std::vector<uchar>& status1,
                                 std::vector<uchar>& status2,
                                 std::vector<uchar>& status3,
                                 std::vector<int>& ages) {
    // getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    for (int i = 0; i < ages.size(); ++i)
    {
        ages[i] += 1;
    }

    int indexCorrection = 0;
    for (int i = 0; i < status3.size(); i++)
    {
        cv::Point2f pt0 = points0.at(i - indexCorrection);
        cv::Point2f pt1 = points1.at(i - indexCorrection);
        cv::Point2f pt2 = points2.at(i - indexCorrection);
        cv::Point2f pt3 = points3.at(i - indexCorrection);
        cv::Point2f pt0_r = points0_return.at(i - indexCorrection);

        if ((status3.at(i) == 0) || (pt3.x < 0) || (pt3.y < 0) || (status2.at(i) == 0) || (pt2.x < 0) || (pt2.y < 0) ||
            (status1.at(i) == 0) || (pt1.x < 0) || (pt1.y < 0) || (status0.at(i) == 0) || (pt0.x < 0) || (pt0.y < 0))
        {
            if ((pt0.x < 0) || (pt0.y < 0) || (pt1.x < 0) || (pt1.y < 0) || (pt2.x < 0) || (pt2.y < 0) || (pt3.x < 0) ||
                (pt3.y < 0))
            {
                status3.at(i) = 0;
            }
            points0.erase(points0.begin() + (i - indexCorrection));
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            points3.erase(points3.begin() + (i - indexCorrection));
            points0_return.erase(points0_return.begin() + (i - indexCorrection));

            ages.erase(ages.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}


void circularMatching(cv::Mat& img_l_0,
                      cv::Mat& img_r_0,
                      cv::Mat& img_l_1,
                      cv::Mat& img_r_1,
                      std::vector<cv::Point2f>& points_l_0,
                      std::vector<cv::Point2f>& points_r_0,
                      std::vector<cv::Point2f>& points_l_1,
                      std::vector<cv::Point2f>& points_r_1,
                      std::vector<cv::Point2f>& points_l_0_return,
                      FeatureSet& current_features) {
    // this function automatically gets rid of points for which tracking fails
    auto t = Timer::set();

    std::vector<float> err;
    cv::Size winSize = cv::Size(21, 21);
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;

    cv::calcOpticalFlowPyrLK(img_l_0, img_r_0, points_l_0, points_r_0, status0, err, winSize, 3, termcrit, 0, 0.001);
    cv::calcOpticalFlowPyrLK(img_r_0, img_r_1, points_r_0, points_r_1, status1, err, winSize, 3, termcrit, 0, 0.001);
    cv::calcOpticalFlowPyrLK(img_r_1, img_l_1, points_r_1, points_l_1, status2, err, winSize, 3, termcrit, 0, 0.001);
    cv::calcOpticalFlowPyrLK(img_l_1, img_l_0, points_l_1, points_l_0_return, status3, err, winSize, 3, termcrit, 0, 0.001);

    deleteUnmatchFeaturesCircle(points_l_0, points_r_0, points_r_1, points_l_1, points_l_0_return, status0, status1,
                                status2, status3, current_features.ages);

    std::cout << "-T circularMatching() time: " << Timer::get(t).count() << "ms" << std::endl;
}


void checkValidMatch(const std::vector<cv::Point2f>& points,
                     const std::vector<cv::Point2f>& points_return,
                     std::vector<bool>& status,
                     const float threshold) {
    status.reserve(points.size());
    for (int i = 0; i < points.size(); ++i)
    {
        const float offset =
            std::max(std::abs(points[i].x - points_return[i].x), std::abs(points[i].y - points_return[i].y));
        if (offset > threshold)
        {
            status.push_back(false);
        }
        else
        {
            status.push_back(true);
        }
    }
}


void removeInvalidPoints(std::vector<cv::Point2f>& points, const std::vector<bool>& status) {
    assert(points.size() == status.size());

    std::vector<cv::Point2f> ret;
    ret.reserve(points.size());
    for (int i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            ret.push_back(points[i]);
        }
    }
    points = std::move(ret);
}


void matchingFeatures(cv::Mat& imageLeft_t0,
                      cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1,
                      cv::Mat& imageRight_t1,
                      FeatureSet& currentVOFeatures,
                      std::vector<cv::Point2f>& pointsLeft_t0,
                      std::vector<cv::Point2f>& pointsRight_t0,
                      std::vector<cv::Point2f>& pointsLeft_t1,
                      std::vector<cv::Point2f>& pointsRight_t1,
                      Map& map,
                      const Config& config)
{
    // Choose detector
    static cv::Ptr<cv::FeatureDetector> detector;
    if (!detector) {
        if (config.use_orb) {
            // Parameters that are not set via the configuration file
            int edge_threshold = config.orb_params.patch_size;
            int first_level = 0;
            int WTA_K = 4;
            cv::ORB::ScoreType score_type = cv::ORB::FAST_SCORE;

            detector = cv::ORB::create(
                config.orb_params.nfeatures,
                config.orb_params.scale_factor,
                config.orb_params.pyr_levels,
                edge_threshold,
                first_level,
                WTA_K,
                score_type,
                config.orb_params.patch_size,
                config.orb_params.fast_treshold
            );
        } else {
            detector = cv::FastFeatureDetector::create(
                config.fast_params.threshold,
                config.fast_params.nonMaxSuppression
            );
        }
    }

    // todo set this value in more proper way
    if (currentVOFeatures.size() < 60)
    {
        // detect new features
        std::vector<cv::KeyPoint> keypoints;
        std::vector<cv::Point2f> new_features;
        auto t = Timer::set();
        detector->detect(imageLeft_t0, keypoints);
        std::cout << "-T Detection time: " << Timer::get(t).count() << "ms" << std::endl;
        cv::KeyPoint::convert(keypoints, new_features, std::vector<int>());

        // append new features with old features
        currentVOFeatures.appendNewFeatures(new_features);
        std::cout << "-- Current feature set size: " << currentVOFeatures.points.size() << std::endl;

        // end current key frame, start new
        map.endKeyFrame();
    }

    const int bucket_size = 64;
    const int features_per_bucket = 4;
    currentVOFeatures.bucketingFeatures(imageLeft_t0, bucket_size, features_per_bucket);

    pointsLeft_t0 = currentVOFeatures.points;

    std::vector<cv::Point2f> pointsLeftReturn_t0; // feature points to check cicular mathcing validation
    circularMatching(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1, pointsLeft_t0, pointsRight_t0,
                     pointsLeft_t1, pointsRight_t1, pointsLeftReturn_t0, currentVOFeatures);

    std::vector<bool> status;
    checkValidMatch(pointsLeft_t0, pointsLeftReturn_t0, status, 1.0f);

    removeInvalidPoints(pointsLeft_t0, status);
    removeInvalidPoints(pointsLeft_t1, status);
    removeInvalidPoints(pointsRight_t0, status);
    removeInvalidPoints(pointsRight_t1, status);

    currentVOFeatures.points = pointsLeft_t1;
}


void trackingFrame2Frame(const cv::Mat& projMatrl,
                         const cv::Mat& projMatrr,
                         std::vector<cv::Point2f>& pointsLeft_t0,
                         std::vector<cv::Point2f>& pointsLeft_t1,
                         cv::Mat& points3D_t0,
                         cv::Matx33d& rotation,
                         cv::Matx31d& translation)
{
    // Calculate frame to frame transformation

    // -----------------------------------------------------------
    // Rotation(R) estimation using Nister's Five Points Algorithm
    // -----------------------------------------------------------
    const double focal = projMatrl.at<float>(0, 0);
    const cv::Point2d principle_point(projMatrl.at<float>(0, 2), projMatrl.at<float>(1, 2));

    // recovering the pose and the essential cv::matrix
    cv::Mat E, mask;
    cv::Mat translation_mono = cv::Mat::zeros(3, 1, CV_64F);
    E = cv::findEssentialMat(pointsLeft_t1, pointsLeft_t0, focal, principle_point, cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, pointsLeft_t1, pointsLeft_t0, rotation, translation_mono, focal, principle_point, mask);
    // std::cout << "recoverPose rotation: " << rotation << std::endl;

    // ------------------------------------------------
    // Translation (t) estimation by use solvePnPRansac
    // ------------------------------------------------
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
    cv::Mat inliers;
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat intrinsic_matrix =
        (cv::Mat_<float>(3, 3) << projMatrl.at<float>(0, 0), projMatrl.at<float>(0, 1), projMatrl.at<float>(0, 2),
         projMatrl.at<float>(1, 0), projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2), projMatrl.at<float>(1, 1),
         projMatrl.at<float>(1, 2), projMatrl.at<float>(1, 3));

    int iterationsCount = 100; // number of Ransac iterations.
    float reprojectionError = 2.0; // maximum allowed distance to consider it an inlier.
    float confidence = 0.99; // RANSAC successful confidence.
    bool useExtrinsicGuess = true;
    int flags = cv::SOLVEPNP_ITERATIVE;

    cv::solvePnPRansac(points3D_t0, pointsLeft_t1, intrinsic_matrix, distCoeffs, rvec, translation, useExtrinsicGuess,
                       iterationsCount, reprojectionError, confidence, inliers, flags);

    translation = -translation;
//    std::cout << "-- inliers size: " << inliers.size() << std::endl;
}


void distinguishNewPoints(std::vector<cv::Point2f>& newPoints,
                          std::vector<bool>& valid,
                          Map& map,
                          size_t frameId_t0,
                          cv::Mat& points3DFrame_t0,
                          cv::Mat& points3DFrame_t1,
                          cv::Mat& points3DWorld,
                          std::vector<cv::Point2f>& currentPointsLeft_t0,
                          std::vector<cv::Point2f>& currentPointsLeft_t1,
                          std::vector<FeaturePoint>& currentFeaturePointsLeft,
                          std::vector<FeaturePoint>& oldFeaturePointsLeft)
{
    // remove exist points, find new points
    // int idx = mapPoints.size();
    currentFeaturePointsLeft.clear();

    for (int i = 0; i < currentPointsLeft_t0.size(); ++i)
    {
        bool exist = false;
        for (auto& oldPoint : oldFeaturePointsLeft)
        {
            if ((oldPoint.mPosOnFrame.x == currentPointsLeft_t0[i].x) &&
                (oldPoint.mPosOnFrame.y == currentPointsLeft_t0[i].y))
            {
                exist = true;

                currentFeaturePointsLeft.emplace_back(oldPoint.ID, oldPoint.mAge + 1, currentPointsLeft_t1[i]);

                Eigen::Vector3f pointPoseIn_t1 = { points3DFrame_t1.at<float>(i, 0),
                                                   points3DFrame_t1.at<float>(i, 1),
                                                   points3DFrame_t1.at<float>(i, 2) };

                map.getPoint(oldPoint.ID)
                    ->addObservation({ frameId_t0, pointPoseIn_t1 });
                break;
            }
        }
        if (!exist)
        {
            // add new points to map points
            Eigen::Vector3f worldPose = { points3DWorld.at<float>(i, 0),
                                          points3DWorld.at<float>(i, 1),
                                          points3DWorld.at<float>(i, 2) };

            // observation from frame t0
            Observation obs1 {
                .mFrameId = frameId_t0,
                .mPointPoseInFrame = { points3DFrame_t0.at<float>(i, 0),
                                       points3DFrame_t0.at<float>(i, 1),
                                       points3DFrame_t0.at<float>(i, 2) }
            };

            // observation from frame t1
            Observation obs2 {
                .mFrameId = frameId_t0 + 1,
                .mPointPoseInFrame = { points3DFrame_t1.at<float>(i, 0),
                                       points3DFrame_t1.at<float>(i, 1),
                                       points3DFrame_t1.at<float>(i, 2) }
            };

            MapPoint* mp = map.createMapPoint(worldPose, { obs1, obs2 });

            newPoints.push_back(currentPointsLeft_t1[i]);

            // add new points to currentFeaturePointsLeft
            currentFeaturePointsLeft.emplace_back(mp->ID, 1, currentPointsLeft_t1[i]);
        }
        valid.push_back(!exist);
    }
}


void integrateOdometryStereo(int frame_i,
                             cv::Mat& rigid_body_transformation,
                             cv::Mat& frame_pose,
                             const cv::Matx33d& rotation,
                             const cv::Matx31d& translation_stereo)
{
    // std::cout << "rotation" << rotation << std::endl;
    // std::cout << "translation_stereo" << translation_stereo << std::endl;

    cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

    cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
    cv::vconcat(rigid_body_transformation, addup, rigid_body_transformation);

    // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;

    double scale = sqrt((translation_stereo(0)) * (translation_stereo(0)) +
                        (translation_stereo(1)) * (translation_stereo(1)) +
                        (translation_stereo(2)) * (translation_stereo(2)));

    // frame_pose = frame_pose * rigid_body_transformation;
//    std::cout << "scale: " << scale << std::endl;

    // rigid_body_transformation = rigid_body_transformation.inv();
    // if ((scale>0.1)&&(translation_stereo.at<double>(2) > translation_stereo.at<double>(0)) &&
    // (translation_stereo.at<double>(2) > translation_stereo.at<double>(1))) if (scale > 0. && scale < 1) { std::cout
    // << "Rpose" << Rpose << std::endl;

    frame_pose = frame_pose * rigid_body_transformation;
    // } else {
    // std::cout << "[WARNING] scale below 0.1, or incorrect translation" << std::endl;
    // }
}
