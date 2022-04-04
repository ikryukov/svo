#include "solve_ba.h"
#include "keyframe.h"
#include "map.h"

#include <ceres/ceres.h>


//////////////////////////////////////////////////////////////////
// math functions needed for rotation conversion.

// dot and cross production

template<typename T>
inline T DotProduct(const T x[3], const T y[3]) {
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

template<typename T>
inline void CrossProduct(const T x[3], const T y[3], T result[3]) {
    result[0] = x[1] * y[2] - x[2] * y[1];
    result[1] = x[2] * y[0] - x[0] * y[2];
    result[2] = x[0] * y[1] - x[1] * y[0];
}


//////////////////////////////////////////////////////////////////

template<typename T>
inline void AngleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3]) {
    const T theta2 = DotProduct(angle_axis, angle_axis);
    if (theta2 > T(std::numeric_limits<double>::epsilon())) {
        // Away from zero, use the rodriguez formula
        //
        //   result = pt costheta +
        //            (w x pt) * sintheta +
        //            w (w . pt) (1 - costheta)
        //
        // We want to be careful to only evaluate the square root if the
        // norm of the angle_axis vector is greater than zero. Otherwise
        // we get a division by zero.
        //
        const T theta = sqrt(theta2);
        const T costheta = cos(theta);
        const T sintheta = sin(theta);
        const T theta_inverse = 1.0 / theta;

        const T w[3] = { angle_axis[0] * theta_inverse,
                         angle_axis[1] * theta_inverse,
                         angle_axis[2] * theta_inverse };

        // Explicitly inlined evaluation of the cross product for
        // performance reasons.
        /*const T w_cross_pt[3] = { w[1] * pt[2] - w[2] * pt[1],
                                  w[2] * pt[0] - w[0] * pt[2],
                                  w[0] * pt[1] - w[1] * pt[0] };*/
        T w_cross_pt[3];
        CrossProduct(w, pt, w_cross_pt);

        const T tmp = DotProduct(w, pt) * (T(1.0) - costheta);
        //    (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T(1.0) - costheta);

        result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
        result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
        result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
    } else {
        // Near zero, the first order Taylor approximation of the rotation
        // matrix R corresponding to a vector w and angle w is
        //
        //   R = I + hat(w) * sin(theta)
        //
        // But sintheta ~ theta and theta * w = angle_axis, which gives us
        //
        //  R = I + hat(w)
        //
        // and actually performing multiplication with the point pt, gives us
        // R * pt = pt + w x pt.
        //
        // Switching to the Taylor expansion near zero provides meaningful
        // derivatives when evaluated using Jets.
        //
        // Explicitly inlined evaluation of the cross product for
        // performance reasons.
        /*const T w_cross_pt[3] = { angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
                                  angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
                                  angle_axis[0] * pt[1] - angle_axis[1] * pt[0] };*/
        T w_cross_pt[3];
        CrossProduct(angle_axis, pt, w_cross_pt);

        result[0] = pt[0] + w_cross_pt[0];
        result[1] = pt[1] + w_cross_pt[1];
        result[2] = pt[2] + w_cross_pt[2];
    }
}

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y)
        : observed_x(observation_x)
        , observed_y(observation_y)
    {}

    template<typename T>
    bool operator()(const T *const rvec,
                    const T *const tvec,
                    const T *const intrinsics,
                    const T *const point,
                    T *residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(rvec, tvec, intrinsics, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *const rvec,
                                                   const T *const tvec,
                                                   const T *const intrinsics,
                                                   const T *const point,
                                                   T* predictions)
    {
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint(rvec, point, p);
        // camera[3,4,5] are the translation
        p[0] += tvec[0];
        p[1] += tvec[1];
        p[2] += tvec[2];

        // Compute the center fo distortion
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion
//        const T& l1 = intrinsics[1];
//        const T& l2 = intrinsics[2];

//        T r2 = xp * xp + yp * yp;
//        T distortion = T(1.0) + r2 * (T(1) + T(1) * r2);
//        T distortion = T(1.0) + r2 * r2;

        const T& focal = intrinsics[0];
        predictions[0] = focal * xp;
        predictions[1] = focal * yp;

        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3, 3, 3, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};


void solveBACeres(const Map& map, KeyFrame* kf) {
    ceres::Problem problem;

    for (auto* mp : kf->mMapPoints) {
        for (auto& obs : mp->mObservations) {
            ceres::CostFunction* costFunction = SnavelyReprojectionError::Create(obs.mPointPoseInFrame.x(),
                                                                                 obs.mPointPoseInFrame.y());
            ceres::LossFunction* lossFunction = new ceres::HuberLoss(1.0);

            Camera* camera = kf->getCamera(obs.mFrameId);
            if (!camera) camera = map.getCamera(obs.mFrameId);

            if (camera) {
                problem.AddResidualBlock(
                    costFunction,
                    lossFunction,
                    camera->rotationEuler().data(),
                    camera->translation().data(),
                    reinterpret_cast<double*>(&Camera::Intrinsics),
                    mp->mWorldPos.data()
                );
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
}
