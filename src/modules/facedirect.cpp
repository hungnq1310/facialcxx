#include "facedirect.h"


FaceDirect::FaceDirect() {        
}

FaceDirect::~FaceDirect() {    
    
}

Point _find_altitude(const Point& A, const Point& B, const Point& C) {
    double x1 = A.x, y1 = A.y;
    double x2 = B.x, y2 = B.y;
    double xp = C.x, yp = C.y;

    double x12 = x2 - x1;
    double y12 = y2 - y1;
    double dotp = x12 * (xp - x1) + y12 * (yp - y1);
    double dot12 = x12 * x12 + y12 * y12;

    if (dot12) {
        double coeff = dotp / dot12;
        double lx = x1 + x12 * coeff;
        double ly = y1 + y12 * coeff;
        return Point(lx, ly);
    } else {
        if (xp - x1 > xp - x2) {
            return Point(xp, y2);
        } else {
            return Point(xp, y1);
        }
    }
}


Point _find_perpendicular(const Point& A, const Point& H, const Point& eye_left, const Point& eye_right, const string& direction) {
    // Calculate the slope of line AH
    double slope_AH = min(-1, H.y - A.y) / max(1, H.x - A.x);

    // Calculate the negative reciprocal of the slope for the perpendicular line AM
    double slope_AM = -1 / slope_AH;

    // Calculate the y-intercept of line AM that passes through point A
    double yAM = A.y - slope_AM * A.x;

    // Calculate the x of point M where line AM intersects the vertical line passing through H
    double xM;
    if (direction == "left") {
        xM = H.x - (abs(eye_right.x - H.x) - abs(eye_left.x - H.x));
    } else {
        xM = H.x + (abs(eye_left.x - H.x) - abs(eye_right.x - H.x));
    }

    // Calculate the y of point M using the x and the equation of line AM
    double yM = slope_AM * xM + yAM;

    // Create a Point with the calculated coordinates
    return Point(xM, yM);
}

void FaceDirect::vis(Mat& img, const Point eye_left, const Point eye_right, const Point nose) {    
    // Point eye_left(det[5], det[6]);
    // Point eye_right(det[7], det[8]);
    // Point nose(det[9], det[10]);

    // Point mount_left(det[11], det[12]);
    // Point mount_right(det[13], det[14]);

    // Calculate coordinates of altitude AH on BC
    Point H = _find_altitude(eye_left, eye_right, nose);
    // Point H_mount = find_altitude(mount_left, mount_right, nose);

    string direction = "right";
    if (abs(eye_left.x - H.x) < abs(eye_right.x - H.x)) {
        direction = "left";
    }

    Point M = _find_perpendicular(nose, H, eye_left, eye_right, direction);

    // double angle = norm(H - nose) / 2 - norm(H_mount - nose);
    // cout << angle << endl;

    line(img, eye_left, eye_right, Scalar(255, 0, 255), 1);
    line(img, eye_left, nose, Scalar(255, 0, 255), 1);
    line(img, eye_right, nose, Scalar(255, 0, 255), 1);
    line(img, H, nose, Scalar(0, 0, 255), 2);
    arrowedLine(img, nose, M, Scalar(255, 0, 0), 2);

    // Uncomment the following lines to draw circles around points
    // circle(img, eye_left, 3, Scalar(0, 255, 0), -1);
    // circle(img, eye_right, 3, Scalar(0, 255, 0), -1);
    // circle(img, nose, 3, Scalar(0, 255, 0), -1);

    putText(img, to_string(abs(eye_left.x - H.x)), Point(eye_left.x - 10, eye_left.y), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0), 1, LINE_AA);
    putText(img, to_string(abs(eye_right.x - H.x)), Point(eye_right.x + 5, eye_right.y), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0), 1, LINE_AA);
    putText(img, to_string(abs(M.x - nose.x)), Point(M.x + 10, M.y), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0), 1, LINE_AA);
    
}