//
// Created by Alexey Klimov on 04.02.2022.
//

#include "drawer.h"

#include <pangolin/display/display.h>
#include <pangolin/gl/opengl_render_state.h>
#include <pangolin/handler/handler.h>
#include <pangolin/display/view.h>

#include "map_point.h"

void drawRectange(const cv::Point3f& center/*, const cv::Mat& rotation*/) {
    float vertices[3][3] = {{center.x-0.5f, center.y-0.5f, center.z+0.0f},
                             {center.x+0.5f, center.y-0.5f, center.z+0.0f},
                             {center.x+0.0f, center.y+0.5f, center.z+0.0f}};

   //cv::Mat M = cv::Mat(3, 3, CV_64F, vertices).inv();

    // This buffer contains floating point vertices with 3 dimensions.
    // They starts from the 0th element and are packed without padding.
    glVertexPointer(3, GL_FLOAT, 0, vertices);

    // Use Them!
    glEnableClientState(GL_VERTEX_ARRAY);

    // Connect the first 3 of these vertices to form a triangle!
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Disable the stuff we enabled...
    glDisableClientState(GL_VERTEX_ARRAY);
}

void Drawer::drawCubeAt(const cv::Point3f& center, float edgeLength) {
    const float e = edgeLength / 2;

    const float cx = center.x;
    const float cy = center.y;
    const float cz = center.z;

    const GLfloat verts[] {
        cx-e, cy-e, cz-e, cx-e, cy+e, cz-e, cx-e, cy+e, cz+e, cx-e, cy-e, cz+e,
        cx+e, cy-e, cz-e, cx+e, cy+e, cz-e, cx+e, cy+e, cz+e, cx+e, cy-e, cz+e,
        cx-e, cy+e, cz-e, cx+e, cy+e, cz-e, cx-e, cy+e, cz+e, cx+e, cy+e, cz+e,
        cx-e, cy-e, cz-e, cx+e, cy-e, cz-e, cx-e, cy-e, cz+e, cx+e, cy-e, cz+e
    };

    glVertexPointer(3, GL_FLOAT, 0, verts);
    glEnableClientState(GL_VERTEX_ARRAY);

    glDrawArrays(GL_LINE_LOOP, 0, 4);
    glDrawArrays(GL_LINE_LOOP, 4, 4);
    glDrawArrays(GL_LINES, 8, 2);
    glDrawArrays(GL_LINES, 10, 2);
    glDrawArrays(GL_LINES, 12, 2);
    glDrawArrays(GL_LINES, 14, 2);

    glDisableClientState(GL_VERTEX_ARRAY);
}

void Drawer::drawMapPoints() {
    glColor4f(0., 0., 1., 1.0);
    glVertexPointer(3, GL_FLOAT, 0, mMapPoints.data());
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, mMapPoints.size() / 3);
    glDisableClientState(GL_VERTEX_ARRAY);
}

void Drawer::drawTrajectory() {
    for (size_t i = 0; i < poses.size(); i++)
    {
        // draw three axes of each pose
        Eigen::Vector3d Ow = poses[i].translation();
        Eigen::Vector3d Xw = poses[i] * (0.1 * Eigen::Vector3d(1, 0, 0));
        Eigen::Vector3d Yw = poses[i] * (0.1 * Eigen::Vector3d(0, 1, 0));
        Eigen::Vector3d Zw = poses[i] * (0.1 * Eigen::Vector3d(0, 0, 1));
        glBegin(GL_LINES);
        glColor3f(1.0, 0.0, 0.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Xw[0], Xw[1], Xw[2]);
        glColor3f(0.0, 1.0, 0.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Yw[0], Yw[1], Yw[2]);
        glColor3f(0.0, 0.0, 1.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Zw[0], Zw[1], Zw[2]);
        glEnd();
    }
    // draw a connection
    for (size_t i = 1; i < poses.size(); i++)
    {
        glColor3f(0.0, 0.0, 0.0);
        glBegin(GL_LINES);
        auto p1 = poses[i-1], p2 = poses[i];
        glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
        glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
        glEnd();
    }
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
    static size_t lastsize = 0;
    std::unique_lock lock(mMutex);

    for (size_t i = lastsize; i < mapPoints.size(); ++i) {
        mMapPoints.push_back(mapPoints[i].mWorldPos.at<float>(0, 0));
        mMapPoints.push_back(mapPoints[i].mWorldPos.at<float>(1, 0));
        mMapPoints.push_back(mapPoints[i].mWorldPos.at<float>(2, 0));
    }

    lastsize = mapPoints.size();
}

void Drawer::addCurrentPose(const Eigen::Isometry3d& quaternion) {
    {
        std::unique_lock lock(mMutex);
    }
    poses.push_back(quaternion);
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
            std::unique_lock lock(drawer->mMutex);
            drawer->drawMapPoints();
            //drawer->drawCurrentPose();
            drawer->drawTrajectory();
            //drawer->drawAllPoses();
        }

        pangolin::FinishFrame();
    }
}
