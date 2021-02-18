#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

#define KITTI

#define MAX_FRAME 4540
#define MIN_NUM_FEAT 2000

bool loadMonoImage(const char* filePattern, int folderIdx, int fileIdx, cv::Mat& outImg)
{
    char fileName[256] = {};
    sprintf(fileName, filePattern, folderIdx, fileIdx);
    outImg = cv::imread(fileName);
    cvtColor(outImg, outImg, COLOR_BGR2GRAY);
    bool res = outImg.data ? true : false;
    return res;
}

bool loadStereoImages(const char* filePattern, int fileIdx, cv::Mat& outImgLeft, cv::Mat& outImgRight)
{
    bool res = false;
    res = loadMonoImage(filePattern, 2, fileIdx, outImgLeft);
    res &= loadMonoImage(filePattern, 3, fileIdx, outImgRight);
    return res;
}

bool loadImages(int fileIdx, cv::Mat& outImgLeft, cv::Mat& outImgRight)
{
    const char* FILE_PATTERN = "/Volumes/IKryukovHDD/data_odometry_color/dataset/sequences/00/image_%d/%06d.png";
    bool res = false;
    res = loadStereoImages(FILE_PATTERN, fileIdx, outImgLeft, outImgRight);
    return res;
}

struct Observation
{
    int frame_id;
    cv::Mat pointPoseInFrame;
};

class MapPoint
{
public:
    // MapPoint();
    MapPoint(int id, cv::Mat worldPos)
    {
        mId = id;
        mWorldPos = worldPos;
    }

    ~MapPoint()
    {
    }

    void addObservation(Observation observation)
    {
        mObservations.push_back(observation);
    }
    int mId;

    // Position in absolute coordinates
    cv::Mat mWorldPos;

    // std::map<Frame*, size_t> mObservations;
    std::vector<Observation> mObservations;
};

struct FeaturePoint
{
    cv::Point2f point;
    int id;
    int age;
};

struct FeatureSet
{
    std::vector<cv::Point2f> points;
    std::vector<int> ages;
    int size()
    {
        return points.size();
    }
    void clear()
    {
        points.clear();
        ages.clear();
    }
};

class Bucket
{
public:
    int id;
    int max_size;

    FeatureSet features;

    Bucket(int size) : max_size(size){};
    ~Bucket()
    {
    }

    void add_feature(cv::Point2f point, int age)
    {
        // won't add feature with age > 10;
        int age_threshold = 10;
        if (age < age_threshold)
        {
            // insert any feature before bucket is full
            if (size() < max_size)
            {
                features.points.push_back(point);
                features.ages.push_back(age);
            }
            else
            // insert feature with old age and remove youngest one
            {
                int age_min = features.ages[0];
                int age_min_idx = 0;

                for (int i = 0; i < size(); i++)
                {
                    if (age < age_min)
                    {
                        age_min = age;
                        age_min_idx = i;
                    }
                }
                features.points[age_min_idx] = point;
                features.ages[age_min_idx] = age;
            }
        }
    }
    void get_features(FeatureSet& current_features)
    {
        current_features.points.insert(current_features.points.end(), features.points.begin(), features.points.end());
        current_features.ages.insert(current_features.ages.end(), features.ages.begin(), features.ages.end());
    }
    int size()
    {
        return features.points.size();
    }
};

void featureDetectionFast(cv::Mat& image, std::vector<cv::Point2f>& points)
{
    // uses FAST as for feature dection, modify parameters as necessary
    std::vector<cv::KeyPoint> keypoints;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    cv::FAST(image, keypoints, fast_threshold, nonmaxSuppression);
    cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}

void appendNewFeatures(cv::Mat& image, FeatureSet& current_features)
{
    std::vector<cv::Point2f> points_new;
    featureDetectionFast(image, points_new);
    current_features.points.insert(current_features.points.end(), points_new.begin(), points_new.end());
    std::vector<int> ages_new(points_new.size(), 0);
    current_features.ages.insert(current_features.ages.end(), ages_new.begin(), ages_new.end());
}

void bucketingFeatures(cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket)
{
    // This function buckets features
    // image: only use for getting dimension of the image
    // bucket_size: bucket size in pixel is bucket_size*bucket_size
    // features_per_bucket: number of selected features per bucket
    int image_height = image.rows;
    int image_width = image.cols;
    int buckets_nums_height = image_height / bucket_size;
    int buckets_nums_width = image_width / bucket_size;
    int buckets_number = buckets_nums_height * buckets_nums_width;

    std::vector<Bucket> Buckets;
    Buckets.reserve((buckets_nums_height + 1) * (buckets_nums_width + 1));
    // initialize all the buckets
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
        for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
        {
            Buckets.push_back(Bucket(features_per_bucket));
        }
    }

    // bucket all current features into buckets by their location
    int buckets_nums_height_idx, buckets_nums_width_idx, buckets_idx;
    for (int i = 0; i < current_features.points.size(); ++i)
    {
        buckets_nums_height_idx = current_features.points[i].y / bucket_size;
        buckets_nums_width_idx = current_features.points[i].x / bucket_size;
        buckets_idx = buckets_nums_height_idx * buckets_nums_width + buckets_nums_width_idx;
        Buckets[buckets_idx].add_feature(current_features.points[i], current_features.ages[i]);
    }

    // get features back from buckets
    current_features.clear();
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
        for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
        {
            buckets_idx = buckets_idx_height * buckets_nums_width + buckets_idx_width;
            Buckets[buckets_idx].get_features(current_features);
        }
    }

    std::cout << "current features number after bucketing: " << current_features.size() << std::endl;
}

void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0,
                                 std::vector<cv::Point2f>& points1,
                                 std::vector<cv::Point2f>& points2,
                                 std::vector<cv::Point2f>& points3,
                                 std::vector<cv::Point2f>& points0_return,
                                 std::vector<uchar>& status0,
                                 std::vector<uchar>& status1,
                                 std::vector<uchar>& status2,
                                 std::vector<uchar>& status3,
                                 std::vector<int>& ages)
{
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
                      FeatureSet& current_features)
{
    // this function automatically gets rid of points for which tracking fails

    std::vector<float> err;
    cv::Size winSize = cv::Size(21, 21);
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;

    calcOpticalFlowPyrLK(img_l_0, img_r_0, points_l_0, points_r_0, status0, err, winSize, 3, termcrit, 0, 0.001);
    calcOpticalFlowPyrLK(img_r_0, img_r_1, points_r_0, points_r_1, status1, err, winSize, 3, termcrit, 0, 0.001);
    calcOpticalFlowPyrLK(img_r_1, img_l_1, points_r_1, points_l_1, status2, err, winSize, 3, termcrit, 0, 0.001);
    calcOpticalFlowPyrLK(img_l_1, img_l_0, points_l_1, points_l_0_return, status3, err, winSize, 3, termcrit, 0, 0.001);

    deleteUnmatchFeaturesCircle(points_l_0, points_r_0, points_r_1, points_l_1, points_l_0_return, status0, status1,
                                status2, status3, current_features.ages);
}

void checkValidMatch(const std::vector<cv::Point2f>& points,
                     const std::vector<cv::Point2f>& points_return,
                     std::vector<bool>& status,
                     const float threshold)
{
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

void removeInvalidPoints(std::vector<cv::Point2f>& points, const std::vector<bool>& status)
{
    int index = 0;
    for (int i = 0; i < status.size(); ++i)
    {
        if (status[i] == false)
        {
            points.erase(points.begin() + index);
        }
        else
        {
            ++index;
        }
    }
}

void matchingFeatures(cv::Mat& imageLeft_t0,
                      cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1,
                      cv::Mat& imageRight_t1,
                      FeatureSet& currentVOFeatures,
                      std::vector<MapPoint> MapPoints,
                      std::vector<cv::Point2f>& pointsLeft_t0,
                      std::vector<cv::Point2f>& pointsRight_t0,
                      std::vector<cv::Point2f>& pointsLeft_t1,
                      std::vector<cv::Point2f>& pointsRight_t1)
{
    if (currentVOFeatures.size() < 2000)
    {
        // append new features with old features
        appendNewFeatures(imageLeft_t0, currentVOFeatures);
        std::cout << "Current feature set size: " << currentVOFeatures.points.size() << std::endl;
    }

    const int bucket_size = 64;
    const int features_per_bucket = 4;
    bucketingFeatures(imageLeft_t0, currentVOFeatures, bucket_size, features_per_bucket);

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
                         cv::Mat& rotation,
                         cv::Mat& translation)
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
    std::cout << "inliers size: " << inliers.size() << std::endl;
}

void displayTracking(cv::Mat& imageLeft_t1, std::vector<cv::Point2f>& pointsLeft_t0, std::vector<cv::Point2f>& pointsLeft_t1)
{
    // -----------------------------------------
    // Display feature racking
    // -----------------------------------------
    int radius = 2;
    cv::Mat vis;

    cv::cvtColor(imageLeft_t1, vis, COLOR_GRAY2BGR, 3);

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
                          std::vector<FeaturePoint>& oldFeaturePointsLeft)
{
    // remove exist points, find new points
    // int idx = mapPoints.size();
    currentFeaturePointsLeft.clear();

    for (int i = 0; i < currentPointsLeft_t0.size(); ++i)
    {
        bool exist = false;
        for (std::vector<FeaturePoint>::iterator oldPointIter = oldFeaturePointsLeft.begin();
             oldPointIter != oldFeaturePointsLeft.end(); ++oldPointIter)
        {
            if ((oldPointIter->point.x == currentPointsLeft_t0[i].x) &&
                (oldPointIter->point.y == currentPointsLeft_t0[i].y))
            {
                exist = true;

                FeaturePoint featurePoint{ .point = currentPointsLeft_t1[i],
                                           .id = oldPointIter->id,
                                           .age = oldPointIter->age + 1 };
                currentFeaturePointsLeft.push_back(featurePoint);

                cv::Mat pointPoseIn_t1 = (cv::Mat_<float>(3, 1) << points3DFrame_t1.at<float>(i, 0),
                                          points3DFrame_t1.at<float>(i, 1), points3DFrame_t1.at<float>(i, 2));
                Observation obs;
                obs.frame_id = frameId_t0 + 1;
                obs.pointPoseInFrame = pointPoseIn_t1;

                mapPoints[oldPointIter->id].addObservation(obs);
                // std::cout << "!!!!!!!!!!!!!!MapPoint  " << oldPointIter->id << " obs : " <<
                // mapPoints[oldPointIter->id].mObservations.size() << std::endl;

                break;
            }
        }
        if (!exist)
        {
            newPoints.push_back(currentPointsLeft_t1[i]);

            // add new points to currentFeaturePointsLeft
            int pointId = mapPoints.size();
            FeaturePoint featurePoint{ .point = currentPointsLeft_t1[i], .id = pointId, .age = 1 };
            currentFeaturePointsLeft.push_back(featurePoint);
            // idx ++;

            // add new points to map points
            cv::Mat worldPose = (cv::Mat_<float>(3, 1) << points3DWorld.at<float>(i, 0), points3DWorld.at<float>(i, 1),
                                 points3DWorld.at<float>(i, 2));

            MapPoint mapPoint(pointId, worldPose);

            // add observation from frame t0
            cv::Mat pointPoseIn_t0 = (cv::Mat_<float>(3, 1) << points3DFrame_t0.at<float>(i, 0),
                                      points3DFrame_t0.at<float>(i, 1), points3DFrame_t0.at<float>(i, 2));
            Observation obs;
            obs.frame_id = frameId_t0;
            obs.pointPoseInFrame = pointPoseIn_t0;
            mapPoint.addObservation(obs);

            // add observation from frame t1
            cv::Mat pointPoseIn_t1 = (cv::Mat_<float>(3, 1) << points3DFrame_t1.at<float>(i, 0),
                                      points3DFrame_t1.at<float>(i, 1), points3DFrame_t1.at<float>(i, 2));
            obs.frame_id = frameId_t0 + 1;
            obs.pointPoseInFrame = pointPoseIn_t1;
            mapPoint.addObservation(obs);

            mapPoints.push_back(mapPoint);
        }
        valid.push_back(!exist);
    }

    // std::cout << "---------------------------------- "  << std::endl;
    // std::cout << "currentPointsLeft size : " << currentPointsLeft.size() << std::endl;
    // std::cout << "points3DFrame_t0 size : " << points3DFrame_t0.size() << std::endl;
    // std::cout << "points3DFrame_t1 size : " << points3DFrame_t1.size() << std::endl;
    // std::cout << "points3DWorld size : " << points3DWorld.size() << std::endl;

    // for (std::vector<cv::Point2f>::iterator currentPointIter = currentPointsLeft.begin() ; currentPointIter !=
    // currentPointsLeft.end(); ++currentPointIter)
    // {
    //     bool exist = false;
    //     for (std::vector<FeaturePoint>::iterator oldPointIter = oldFeaturePointsLeft.begin() ; oldPointIter !=
    //     oldFeaturePointsLeft.end(); ++oldPointIter)
    //     {
    //         if ((oldPointIter->point.x == currentPointIter->x) && (oldPointIter->point.y == currentPointIter->y))
    //         {
    //            exist = true;

    //            FeaturePoint featurePoint{.point=*currentPointIter, .id=oldPointIter->id};
    //            currentFeaturePointsLeft.push_back(featurePoint);
    //            break;
    //         }
    //     }
    //     if (!exist)
    //     {
    //         newPoints.push_back(*currentPointIter);

    //         FeaturePoint featurePoint{.point=*currentPointIter, .id=idx};
    //         currentFeaturePointsLeft.push_back(featurePoint);
    //         idx ++;

    //     }
    //     valid.push_back(!exist);

    // }
    std::cout << "newPoints size : " << newPoints.size() << std::endl;
}

bool isRotationMatrix(cv::Mat& R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    return norm(I, shouldBeIdentity) < 1e-6;
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat& R)
{
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

void integrateOdometryStereo(int frame_i,
                             cv::Mat& rigid_body_transformation,
                             cv::Mat& frame_pose,
                             const cv::Mat& rotation,
                             const cv::Mat& translation_stereo)
{
    // std::cout << "rotation" << rotation << std::endl;
    // std::cout << "translation_stereo" << translation_stereo << std::endl;

    cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

    cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
    cv::vconcat(rigid_body_transformation, addup, rigid_body_transformation);

    // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;

    double scale = sqrt((translation_stereo.at<double>(0)) * (translation_stereo.at<double>(0)) +
                        (translation_stereo.at<double>(1)) * (translation_stereo.at<double>(1)) +
                        (translation_stereo.at<double>(2)) * (translation_stereo.at<double>(2)));

    // frame_pose = frame_pose * rigid_body_transformation;
    std::cout << "scale: " << scale << std::endl;

    // rigid_body_transformation = rigid_body_transformation.inv();
    // if ((scale>0.1)&&(translation_stereo.at<double>(2) > translation_stereo.at<double>(0)) &&
    // (translation_stereo.at<double>(2) > translation_stereo.at<double>(1))) if (scale > 0. && scale < 1) { std::cout
    // << "Rpose" << Rpose << std::endl;

    frame_pose = frame_pose * rigid_body_transformation;
    // } else {
    // std::cout << "[WARNING] scale below 0.1, or incorrect translation" << std::endl;
    // }
}

void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose, std::vector<Mat>& pose_matrix_gt, float fps, bool show_gt)
{
    // draw estimated trajectory
    int x = int(pose.at<double>(0)) + 300;
    int y = int(pose.at<double>(2)) + 100;
    circle(trajectory, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);

    if (show_gt)
    {
        // draw ground truth trajectory
        cv::Mat pose_gt = cv::Mat::zeros(1, 3, CV_64F);

        // pose_gt.at<double>(0) = pose_matrix_gt[frame_id].val[0][3];
        // pose_gt.at<double>(1) = pose_matrix_gt[frame_id].val[0][7];
        // pose_gt.at<double>(2) = pose_matrix_gt[frame_id].val[0][11];
        // x = int(pose_gt.at<double>(0)) + 300;
        // y = int(pose_gt.at<double>(2)) + 100;
        // circle(trajectory, cv::Point(x, y) ,1, CV_RGB(255,255,0), 2);
    }
    // print info

    // rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    // sprintf(text, "FPS: %02f", fps);
    // putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    cv::imshow("Trajectory", trajectory);

    cv::waitKey(1);
}

int main(int argc, char** argv)
{
    int init_frame_id = 0;

    // ------------------------
    // Load first images
    // ------------------------
    cv::Mat imageLeft_t0;
    cv::Mat imageRight_t0;
    loadImages(init_frame_id, imageLeft_t0, imageRight_t0);

    if (!imageLeft_t0.data || !imageRight_t0.data)
    {
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
    cout << "P_left: " << endl << projMatrl << endl;
    cout << "P_right: " << endl << projMatrr << endl;

    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation_stereo = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat pose = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);

    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat frame_pose32 = cv::Mat::eye(4, 4, CV_32F);

    cv::Point2d pp(cx, cy);
    Matx33d K = Matx33d(focal, 0, cx, 0, focal, cy, 0, 0, 1);

    clock_t begin = clock();

    cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);

    cv::Mat points4D, points3D;
    FeatureSet currentVOFeatures;
    std::vector<MapPoint> mapPoints;

    std::vector<FeaturePoint> oldFeaturePointsLeft;
    std::vector<FeaturePoint> currentFeaturePointsLeft;

    for (int numFrame = init_frame_id + 1; numFrame < MAX_FRAME; ++numFrame)
    {
        cout << numFrame << endl;

        // ------------
        // Load images
        // ------------
        cv::Mat imageLeft_t1;
        cv::Mat imageRight_t1;
        bool res = loadImages(numFrame, imageLeft_t1, imageRight_t1);
        if (!res)
        {
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
        // displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);

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
        cv::Mat pose = frame_pose.col(3).clone();

        std::vector<cv::Mat> pose_matrix_gt;

        display(numFrame, trajectory, pose, pose_matrix_gt, 0.0, false);
    }

    return 0;
}
