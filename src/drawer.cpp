//
// Created by Alexey Klimov on 04.02.2022.
//

#include "drawer.h"

#include <pangolin/display/display.h>
#include <pangolin/gl/opengl_render_state.h>
#include <pangolin/handler/handler.h>
#include <pangolin/display/view.h>

#include "map_point.h"


void Drawer::drawMapPoints() {
    glColor4f(0., 0., 1., 1.0);
    glVertexPointer(3, GL_FLOAT, 0, mMapPoints.data());
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, mMapPoints.size() / 3);
    glDisableClientState(GL_VERTEX_ARRAY);
}

void Drawer::drawTrajectory() {
    glBegin(GL_LINES);
    for (auto& pose : poses)
    {
        // draw three axes of each pose
        Eigen::Vector3d Ow = pose.translation();
        Eigen::Vector3d Xw = pose * (0.1 * Eigen::Vector3d(1, 0, 0));
        Eigen::Vector3d Yw = pose * (0.1 * Eigen::Vector3d(0, 1, 0));
        Eigen::Vector3d Zw = pose * (0.1 * Eigen::Vector3d(0, 0, 1));
        glColor3f(1.0, 0.0, 0.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Xw[0], Xw[1], Xw[2]);
        glColor3f(0.0, 1.0, 0.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Yw[0], Yw[1], Yw[2]);
        glColor3f(0.0, 0.0, 1.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Zw[0], Zw[1], Zw[2]);
    }
    // draw a connection
    for (size_t i = 1; i < poses.size(); i++)
    {
        glColor3f(1.0, 1.0, 1.0);
        auto t1 = poses[i-1].translation(),
             t2 = poses[i].translation();
        glVertex3d(t1[0], t1[1], t1[2]);
        glVertex3d(t2[0], t2[1], t2[2]);
    }
    glEnd();
}

/////////////////////////////////////////////////////// Drawer ///////////////////////////////////////////////////////

Drawer::Drawer()
    : mThread(run, this)
{}

Drawer::~Drawer() {
    mIsFinish = true;
    mThread.join();
}

void Drawer::addMapPoints(const std::vector<MapPoint>& mapPoints) {
    static size_t lastSize = 0;
    size_t newSize = mapPoints.size();
    std::unique_lock lock(mDrawerMutex);

    mMapPoints.reserve(newSize);
    for (size_t i = lastSize; i < newSize; ++i) {
        mMapPoints.push_back(mapPoints[i].mWorldPos.at<float>(0, 0));
        mMapPoints.push_back(mapPoints[i].mWorldPos.at<float>(1, 0));
        mMapPoints.push_back(mapPoints[i].mWorldPos.at<float>(2, 0));
    }

    lastSize = mapPoints.size();
}

void Drawer::addCurrentPose(const cv::Matx<double, 3, 3>& rotation, const cv::Matx<double, 3, 1>& pose) {
    Eigen::Matrix<double, 3, 3> rotationEigen;
    cv::cv2eigen(rotation, rotationEigen);
    Eigen::Vector3d translationEigen{ pose(0), pose(1), pose(2) };
    {
        std::unique_lock lock(mDrawerMutex);
        if (poses.size() > 1) {
            auto p = poses.back();
            p.rotate(rotationEigen);
            p.translate(translationEigen);
            poses.push_back(p);
        } else {
            poses.emplace_back(Eigen::Quaternion<double>::Identity());
        }
    }
}

void Drawer::run(Drawer* drawer) {
    pangolin::CreateWindowAndBind("VO", 640, 480);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                                .SetHandler(&handler);

    while(!pangolin::ShouldQuit() && !drawer->mIsFinish) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        {
            std::unique_lock lock(drawer->mDrawerMutex);
            drawer->drawMapPoints();
            drawer->drawTrajectory();
        }

        pangolin::FinishFrame();
    }
}
