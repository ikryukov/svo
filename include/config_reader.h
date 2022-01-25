//
// Created by Smirnov Grigory on 22.01.2022.
//

#pragma once

#include <string>
#include <iostream>
#include <exception>
#include <opencv2/core.hpp>

struct Config {
    std::string path;
    std::string gt_path;
    double focal;
    double cx;
    double cy;
    double bf;
    int start_frame;
    int end_frame;
    bool show_gt;
};

class ConfigReader {
private:
    Config config;
public:
    ConfigReader(std::string& filename) {
        try {
            cv::FileStorage fs(filename, cv::FileStorage::READ);
            if (fs.isOpened()) {
                fs["path"] >> config.path;
                fs["focal"] >> config.focal;
                fs["cx"] >> config.cx;
                fs["cy"] >> config.cy;
                fs["bf"] >> config.bf;
                fs["start_frame"] >> config.start_frame;
                fs["end_frame"] >> config.end_frame;
                fs["show_gt"] >> config.show_gt;
                fs["gt_path"] >> config.gt_path;
            }
        }
        catch (std::exception& e) {
            std::cout << "Wrong config file!" << std::endl;
            std::cout << e.what();
        }
    }
    const Config& getConfig() const
    {
        return config;
    }
};
