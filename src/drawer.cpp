//
// Created by Alexey Klimov on 04.02.2022.
//

#include "drawer.h"

#include <pangolin/display/display.h>
#include <pangolin/gl/opengl_render_state.h>
#include <pangolin/handler/handler.h>
#include <pangolin/display/view.h>

#include "map.h"


__forceinline void drawMapPoints(const auto& points) {
    glPointSize(2);
    glColor4f(0., 0., 1., 1.0);
    glBegin(GL_POINTS);

    for (const MapPoint* mp : points) {
        glVertex3f(mp->mWorldPos.x(), mp->mWorldPos.y(), mp->mWorldPos.z());
    }

    glEnd();
}

__forceinline void drawTrajectory(const auto& poses) {
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

Drawer::Drawer(const Map& map)
    : mMap(map)
    , mThread([this]() {
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

         while(!pangolin::ShouldQuit() && !mIsFinish) {
             glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
             d_cam.Activate(s_cam);

             {
                 std::shared_lock lock(mMap.mMapMutex);
                 for (auto& kf : mMap.mKeyFrames) {
                     drawMapPoints(kf->mMapPoints);
                     drawTrajectory(kf->mPoses);
                 }
             }

             {
                 std::shared_lock lock(mMap.mCurrentKFMutex);
                 drawMapPoints(mMap.mCurrentKeyFrame->mMapPoints);
                 drawTrajectory(mMap.mCurrentKeyFrame->mPoses);
             }

             pangolin::FinishFrame();
         }
    })
{}

Drawer::~Drawer() {
    mIsFinish = true;
    mThread.join();
}
