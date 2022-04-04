
#pragma once

#include <Eigen/Core>

#include <vector>


class MapPoint {

    friend class Map;

public:

    const size_t ID;

    MapPoint* addObservation(const size_t frameID, const float observed_x, const float observed_y) {
        mObservations.emplace_back(Observation { frameID, { observed_x, observed_y } });
        return this;
    }

    MapPoint* setPosition(const Eigen::Vector3d& pos) {
        mWorldPos = pos;
        return this;
    }

private:

    struct Observation {
        size_t mFrameId;
        Eigen::Vector2f mPointPoseInFrame;
    };

    explicit MapPoint(size_t id)
        : ID(id)
    {}

    MapPoint(size_t id, const Eigen::Vector3d& worldPos)
        : ID(id)
        , mWorldPos(worldPos)
    {}

public:

    Eigen::Vector3d mWorldPos;
    std::vector<Observation> mObservations;
};
