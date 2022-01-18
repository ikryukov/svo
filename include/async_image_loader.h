//
// Created by Alexey Klimov on 13.01.2022.
//

#ifndef ASYNCIMAGELOADER_H_
#define ASYNCIMAGELOADER_H_


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <readerwriterqueue.h>


using moodycamel::ReaderWriterQueue;

class AsyncImageLoader {
public:
    explicit AsyncImageLoader(const char* folder, size_t start_frame,
                              size_t last_frame, bool color = true):
            mQueue(last_frame - start_frame)
          , mDatasetFolder(folder)
          , mIsColor(color)
          , mThread([=]() {
              for(auto i = start_frame; (i < last_frame) && finish; ++i) {
                  cv::Mat l, r;
                  syncLoad(i, l, r);
                  mQueue.emplace(std::move(l), std::move(r));
              }
          })
    {}

    bool get(cv::Mat& left, cv::Mat& right) {
        std::pair<cv::Mat, cv::Mat> temp{};

        while (!mQueue.peek()) {}

        bool res = mQueue.try_dequeue(temp);
        left = std::move(temp.first);
        right = std::move(temp.second);

        return res;
    }

    ~AsyncImageLoader() {
        finish = false;
        mThread.join();
    }

private:
    bool syncLoad(size_t i, cv::Mat& left, cv::Mat& right) {
        size_t f1, f2;
        if (!mIsColor) { f1 = 0; f2 = 1; }
        else           { f1 = 2; f2 = 3; }

        return loadImg(i, f1, left) & loadImg(i, f2, right);
    }

    bool loadImg(size_t i, size_t fIdx, cv::Mat& imgDst) {
        char fileName[128] = {};
        sprintf(fileName, (mDatasetFolder + "image_%d/%06d.png").c_str(), fIdx, i);
        imgDst = cv::imread(fileName);

        if (mIsColor && imgDst.data)
            cv::cvtColor(imgDst, imgDst, cv::COLOR_BGR2GRAY);

        return imgDst.data;
    }

    std::atomic<bool> finish = true;
    ReaderWriterQueue<std::pair<cv::Mat, cv::Mat>> mQueue;
    std::thread mThread;
    const std::string mDatasetFolder;
    bool mIsColor;
};

#endif // ASYNCIMAGELOADER_H_
