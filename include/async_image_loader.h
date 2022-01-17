//
// Created by Alexey Klimov on 13.01.2022.
//

#ifndef ASYNCIMAGELOADER_H_
#define ASYNCIMAGELOADER_H_


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "SPSCQueue.h"

using namespace rigtorp;


class AsyncImageLoader {
public:
    explicit AsyncImageLoader(const char* folder, bool color = true):
            _q(1)
          , _folder(folder)
          , _color(color)
          , _thread([&]() {
              size_t i_ = 0;

              while (_b) {
                  // push a new pair of images
                  cv::Mat l, r;
                  bool res = syncLoad(i_, l, r);
                  i_++;
                  _q.push({std::move(l), std::move(r)});

                  // wait until queue is empty
                  while (_q.front() && _b) {}
              }
          })
    {}

    std::pair<cv::Mat, cv::Mat> get() {
        // wait until something appears in the queue
        while(!_q.front()) {}

        // take it
        auto ret = *_q.front();
        _q.pop();
        return ret;
    }

    ~AsyncImageLoader() {
        _b = false;
        _thread.join();
    }

private:
    bool syncLoad(size_t i, cv::Mat& left, cv::Mat& right) {
        size_t f1, f2;
        if (!_color) { f1 = 0; f2 = 1; }
        else         { f1 = 2; f2 = 3; }

        return loadImg(i, f1, left) & loadImg(i, f2, right);
    }

    bool loadImg(size_t i, size_t fIdx, cv::Mat& imgDst) {
        char fileName[128] = {};
        sprintf(fileName, (_folder + "image_%d/%06d.png").c_str(), fIdx, i);
        imgDst = cv::imread(fileName);

        if (_color && imgDst.data)
            cv::cvtColor(imgDst, imgDst, cv::COLOR_BGR2GRAY);

        return imgDst.data;
    }

    std::atomic<bool> _b = true;
    SPSCQueue<std::pair<cv::Mat, cv::Mat>> _q;
    std::thread _thread;
    const std::string _folder;
    bool _color;
};

#endif // ASYNCIMAGELOADER_H_
