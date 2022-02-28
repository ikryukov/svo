//
// Created by Alexey Klimov on 04.02.2022.
//

#pragma once

#include <thread>
#include <atomic>


class Map;


class Drawer {
public:
    explicit Drawer(Map& map);
    ~Drawer();

private:
    std::atomic<bool> mIsFinish = false;
    Map& mMap;
    std::thread mThread;
};
