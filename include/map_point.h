#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <opencv2/core/mat.hpp>

#include <utility>
#include <vector>


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
        : mId(id)
        , mWorldPos(std::move(worldPos))
    {}

    ~MapPoint() = default;

    void addObservation(const Observation& observation) {
        mObservations.push_back(observation);
    }

private:
    int mId;

    // Position in absolute coordinates
    cv::Mat mWorldPos;

    // std::map<Frame*, size_t> mObservations;
    std::vector<Observation> mObservations;
};

#endif