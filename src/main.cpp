#include <iostream>
#include <opencv2/calib3d.hpp>

#include "async_image_loader.h"
#include "map_point.h"
#include "bucket.h"
#include "utils.h"
#include "core.h"

#define KITTI

#define MAX_FRAME 4540
#define MIN_NUM_FEAT 2000


int main(int argc, char** argv)
{
    int init_frame_id = 0;

    // ------------------------
    // Load first images
    // ------------------------
    // TODO set values via config file. Currently hardcoded for first sequence
    AsyncImageLoader async_image_loader("../../datasets/color/dataset/sequences/00/", 0, MAX_FRAME, MAX_FRAME);
    cv::Mat imageLeft_t0, imageRight_t0;

    if (!async_image_loader.get(imageLeft_t0, imageRight_t0)) {
        std::cout << " --(!) Error reading images " << std::endl;
        return -1;
    }

// TODO: add a fucntion to load these values directly from KITTI's calib files
// WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
#ifdef KITTI
    double focal = 718.8560;
    double cx = 607.1928;
    double cy = 185.2157;
#else
    // iPhone X
    // double focal = 1591.0;// ARkit mode;
    double focal = 28.0 / 36.0 * 1881.0;
    double cx = 1065.0 / 2.0;
    double cy = 1881.0 / 2.0;
#endif
    double fx = focal;
    double fy = focal;
    double bf = -386.1448;
    const cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0, 0., 1., 0.);
    const cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0, 0., 1., 0.);
//    cout << "P_left: " << endl << projMatrl << endl;
//    cout << "P_right: " << endl << projMatrr << endl;

    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation_stereo = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat pose = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);

    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat frame_pose32 = cv::Mat::eye(4, 4, CV_32F);

    cv::Point2d pp(cx, cy);
    cv::Matx33d K = cv::Matx33d(focal, 0, cx, 0, focal, cy, 0, 0, 1);

    cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);

    cv::Mat points4D, points3D;
    FeatureSet currentVOFeatures;
    std::vector<MapPoint> mapPoints;

    std::vector<FeaturePoint> oldFeaturePointsLeft;
    std::vector<FeaturePoint> currentFeaturePointsLeft;

    for (int numFrame = init_frame_id + 1; numFrame < MAX_FRAME; ++numFrame)
    {
        clock_t begin = clock();
        std::cout << std::endl <<  numFrame << std::endl;

        // ------------
        // Load images
        // ------------
        cv::Mat imageLeft_t1, imageRight_t1;

        if (!async_image_loader.get(imageLeft_t1, imageRight_t1)) {
            std::cout << " --(!) Error reading images " << std::endl;
            break;
        }

        std::vector<cv::Point2f> oldPointsLeft_t0 = currentVOFeatures.points;

        std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;

        matchingFeatures(imageLeft_t0, imageRight_t0, imageLeft_t1, imageRight_t1, currentVOFeatures, mapPoints,
                         pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1);

        imageLeft_t0 = imageLeft_t1;
        imageRight_t0 = imageRight_t1;

        std::vector<cv::Point2f>& currentPointsLeft_t0 = pointsLeft_t0;
        std::vector<cv::Point2f>& currentPointsLeft_t1 = pointsLeft_t1;

        std::cout << "oldPointsLeft_t0 size : " << oldPointsLeft_t0.size() << std::endl;
        std::cout << "currentFramePointsLeft size : " << currentPointsLeft_t0.size() << std::endl;

        // ---------------------
        // Triangulate 3D Points
        // ---------------------
        cv::Mat points3D_t0, points4D_t0;
        cv::triangulatePoints(projMatrl, projMatrr, pointsLeft_t0, pointsRight_t0, points4D_t0);
        cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);
        // std::cout << "points4D_t0 size : " << points4D_t0.size() << std::endl;

        cv::Mat points3D_t1, points4D_t1;
        // std::cout << "pointsLeft_t1 size : " << pointsLeft_t1.size() << std::endl;
        // std::cout << "pointsRight_t1 size : " << pointsRight_t1.size() << std::endl;

        cv::triangulatePoints(projMatrl, projMatrr, pointsLeft_t1, pointsRight_t1, points4D_t1);
        cv::convertPointsFromHomogeneous(points4D_t1.t(), points3D_t1);
        // std::cout << "points4D_t1 size : " << points4D_t1.size() << std::endl;

        trackingFrame2Frame(
            projMatrl, projMatrr, pointsLeft_t0, pointsLeft_t1, points3D_t0, rotation, translation_stereo);
        displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);

        points4D = points4D_t0;
        frame_pose.convertTo(frame_pose32, CV_32F);
        points4D = frame_pose32 * points4D;
        cv::convertPointsFromHomogeneous(points4D.t(), points3D);

        std::vector<cv::Point2f> newPoints;
        std::vector<bool> valid;
        distinguishNewPoints(newPoints, valid, mapPoints, numFrame - 1, points3D_t0, points3D_t1, points3D,
                             currentPointsLeft_t0, currentPointsLeft_t1, currentFeaturePointsLeft, oldFeaturePointsLeft);
        oldFeaturePointsLeft = currentFeaturePointsLeft;
        std::cout << "mapPoints size : " << mapPoints.size() << std::endl;

        // ------------------------------------------------
        // Append feature points to Point clouds
        // ------------------------------------------------
        // featureSetToPointCloudsValid(points3D, features_cloud_ptr, valid);
        // mapPointsToPointCloudsAppend(mapPoints, features_cloud_ptr);

        cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);

        cv::Mat rigid_body_transformation;
        if (abs(rotation_euler[1]) < 0.1 && abs(rotation_euler[0]) < 0.1 && abs(rotation_euler[2]) < 0.1)
        {
            integrateOdometryStereo(numFrame, rigid_body_transformation, frame_pose, rotation, translation_stereo);
        }
        else
        {
            std::cout << "Too large rotation" << std::endl;
        }

        Rpose = frame_pose(cv::Range(0, 3), cv::Range(0, 3));
        cv::Vec3f Rpose_euler = rotationMatrixToEulerAngles(Rpose);
        pose = frame_pose.col(3).clone();

        std::vector<cv::Mat> pose_matrix_gt;

        display(numFrame, trajectory, pose, pose_matrix_gt, 0.0, false);
        std::cout << "Frame time: " << static_cast<double>(clock() - begin) / CLOCKS_PER_SEC << std::endl;
    }

    return 0;
}
