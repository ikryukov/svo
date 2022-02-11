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

void Drawer::drawCurrentPose() {
    if (!pose.empty()) {
        glColor4f(1., 0., 0., 1.0);
        cv::Point3d point_3_f { pose.at<double>(0), pose.at<double>(1), pose.at<double>(2) };
        drawCubeAt(point_3_f, 1);
        //drawRectange(point_3_f);
    }
}

void Drawer::drawAllPoses() {
    for (int i=0; i < mTrajectory.size(); i+=3) {
        cv::Point3d point_3_f { mTrajectory[i], mTrajectory[i+1], mTrajectory[i+2] };
        drawCubeAt(point_3_f, 0.2);
//        glBegin(GL_LINES);
//        glVertex3f(point_3_f.x, point_3_f.y, point_3_f.z);
//        glVertex3f(point_3_f.x+mDirection[i], point_3_f.y+mDirection[i+1], point_3_f.z+mDirection[i+2]);
//        glEnd();
    }
}

void Drawer::drawTrajectory() {
    glColor4f(1., 0., 0., 1.0);
    glVertexPointer(3, GL_DOUBLE, 0, mTrajectory.data());
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_LINE_STRIP, 0, mTrajectory.size() / 3);
    glDisableClientState(GL_VERTEX_ARRAY);

//    glColor4f(0., 1., 0., 1.0);
//    glVertexPointer(3, GL_DOUBLE, 0, mDirection.data());
//    glEnableClientState(GL_VERTEX_ARRAY);
//    glDrawArrays(GL_LINE_STRIP, 0, mDirection.size() / 3);
//    glDisableClientState(GL_VERTEX_ARRAY);
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

void Drawer::addCurrentPose(const cv::Mat& p, const cv::Vec3f& r) {
    {
        std::unique_lock lock(mMutex);
        this->pose = p;
        this->rotation_euler = r;
    }
    mTrajectory.push_back(pose.at<double>(0));
    mTrajectory.push_back(pose.at<double>(1));
    mTrajectory.push_back(pose.at<double>(2));

    mDirection.push_back(pose.at<double>(0) + rotation_euler[0]);
    mDirection.push_back(pose.at<double>(1) + rotation_euler[1]);
    mDirection.push_back(pose.at<double>(2) + rotation_euler[2]);
}

void Drawer::run(Drawer* drawer) {
    pangolin::CreateWindowAndBind("VO", 640, 480);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,500),
        pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
                                .SetHandler(&handler);


    while(!pangolin::ShouldQuit() && !drawer->mIsFinish) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        {
            std::unique_lock lock(drawer->mMutex);
            drawer->drawMapPoints();
            //drawer->drawCurrentPose();
            drawer->drawTrajectory();
            drawer->drawAllPoses();
        }

        pangolin::FinishFrame();
    }
}
