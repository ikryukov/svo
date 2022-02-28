//
// Created by Alexey Klimov on 27.02.2022.
//

#include "key_frame.h"


KeyFrame::KeyFrame(const KeyFrame& other)
    : mMapPoints(other.mMapPoints)
    , mPoses(other.mPoses)
{}

KeyFrame::KeyFrame(KeyFrame&& other) noexcept
    : mMapPoints(std::move(other.mMapPoints))
    , mPoses(std::move(other.mPoses))
{}

KeyFrame& KeyFrame::operator=(const KeyFrame& other) {
    if (&other == this) return *this;

    this->mMapPoints = other.mMapPoints;
    this->mPoses = other.mPoses;

    return *this;
}

KeyFrame& KeyFrame::operator=(KeyFrame&& other) noexcept {
    if (&other == this) return *this;

    this->mMapPoints = std::move(other.mMapPoints);
    this->mPoses = std::move(other.mPoses);

    return *this;
}

KeyFrame::~KeyFrame() {
    for (auto* mp : mMapPoints)
        delete mp;
}
