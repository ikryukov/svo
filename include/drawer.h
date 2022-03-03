//
// Created by Alexey Klimov on 04.02.2022.
//

#pragma once

#include <thread>
#include <atomic>


class Map;


class Drawer {
public:
    explicit Drawer(const Map& map);
    ~Drawer();

private:
    std::atomic<bool> mIsFinish = false;
    const Map& mMap;
    std::thread mThread;
};
