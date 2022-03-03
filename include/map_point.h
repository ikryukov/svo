
#pragma once

#include <Eigen/Core>

#include <vector>


struct Observation {
    size_t mFrameId;
    Eigen::Vector3f mPointPoseInFrame;
};


class MapPoint {

    friend class Map;

public:

    const size_t ID;

    MapPoint* addObservation(const Observation& observation) {
        mObservations.push_back(observation);
        return this;
    }

    MapPoint* setPosition(const Eigen::Vector3f& pos) {
        mWorldPos = pos;
        return this;
    }

private:

    explicit MapPoint(size_t id)
        : ID(id)
    {}

    MapPoint(size_t id, const Eigen::Vector3f& worldPos)
        : ID(id)
        , mWorldPos(worldPos)
    {}

    MapPoint(size_t id, const Eigen::Vector3f& worldPos, const std::vector<Observation>& obss)
        : ID(id)
        , mWorldPos(worldPos)
        , mObservations(obss)
    {}

public:

    Eigen::Vector3f mWorldPos;
    std::vector<Observation> mObservations;
};
