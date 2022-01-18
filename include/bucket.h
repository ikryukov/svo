#ifndef BUCKET_H
#define BUCKET_H

#include <opencv2/core/types.hpp>

#include <vector>

void featureDetectionFast(cv::Mat& image, std::vector<cv::Point2f>& points);

struct FeaturePoint
{
    cv::Point2f point;
    int id;
    int age;
};

struct FeatureSet
{
    [[nodiscard]] int size() const;
    void clear();

    void appendNewFeatures(cv::Mat& image);
    void bucketingFeatures(cv::Mat& image, int bucket_size, int features_per_bucket);

    friend class Bucket;

    std::vector<cv::Point2f> points;
    std::vector<int> ages;
};


class Bucket
{
public:
    explicit Bucket(int size) : max_size(size){};
    ~Bucket() = default;

    void add_feature(cv::Point2f point, int age);
    void get_features(FeatureSet& current_features);
    int size();

private:
    int id;
    int max_size;

    FeatureSet features;
};

#endif