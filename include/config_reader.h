//
// Created by Smirnov Grigory on 22.01.2022.
//

#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <string>
#include <iostream>
#include <exception>
#include <opencv2/core.hpp>

class ConfigReader {
private:
    std::string path;
    double focal;
    double cx;
    double cy;
    double bf;
    int start_frame;
    int end_frame;
    bool show_gt;
public:
    ConfigReader(std::string filename) {
        try {
            cv::FileStorage config(filename, cv::FileStorage::READ);
            if (config.isOpened()) {
                config["path"] >> path;
                config["focal"] >> focal;
                config["cx"] >> cx;
                config["cy"] >> cy;
                config["bf"] >> bf;
                config["start_frame"] >> start_frame;
                config["end_frame"] >> end_frame;
                config["show_gt"] >> show_gt;
            }
        }
        catch (std::exception& e) {
            std::cout << "Wrong config file!" << std::endl;
            std::cout << e.what();
        }
    }
    const std::string& getPath() const
    {
        return path;
    }
    double getFocal() const
    {
        return focal;
    }
    double getCx() const
    {
        return cx;
    }
    double getCy() const
    {
        return cy;
    }
    double getBf() const
    {
        return bf;
    }
    int getStartFrame() const
    {
        return start_frame;
    }
    int getEndFrame() const
    {
        return end_frame;
    }
    bool isShowGt() const
    {
        return show_gt;
    }
};

#endif // CONFIG_READER_H
