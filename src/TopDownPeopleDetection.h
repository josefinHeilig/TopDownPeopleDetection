#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

class BlobPerson;

// CLASS TO RECEIVE EACH NEW DEPTH FRAME AND TO DETECT BLOB PERSONS //
class TopDownPeopleDetector {

private:

	// List of all detections of BlobPersons //
	std::vector<BlobPerson> allBlobPersons;

	// Threshold to discard too small contours //
	int thresholdContourArea = 4000; //8000;
	int thresholdContourAreaHead = 1000;//1000;

	// Higher clipping plane, to take all the height to the camera, put 0 //
	int clippingMinDepth = 1000; // 1000; // to avoid values that are 0 because of absorption of material
	// Lower to the ground clipping plane //
	int clippingMaxDepth = 3200; //3400

	// Distance from Camera to floor //
	int realDepth = 3400; //3750 clippingMaxDepth and realDepth normally should be the same, distance  from camera to the floor, but data gets to noisy there

	// CLIP IMAGE ON SIDES TO AVOID THINGS THAT SHOULD NOT BE SEEN, EG. TRIPODS, WALL, ETC IN PIXELS
	int xMin = 120;
	int xMax = 50;
	int yMin = 0;
	int yMax = 20;

	//GET LOCATION AND VALUE OF HEIGHEST VALUE IN MAT //
	void getHeighestValueLocationInMat(cv::Mat& m, cv::Point& maxLoc, double& maxVal);

	// IMAGE TO DRAW DETECTIONS ON //
	cv::Mat outImage;

public:


	// INITIALIZE TOP DOWN PEOPLE DETECTOR //
	TopDownPeopleDetector();

	// UPDATE DETECTION BASED ON DEPTH IMAGE IN CV:MAT FORMAT //
	void update(cv::Mat depthImage);

	// GET BLOBPERSON LIST //
	std::vector<BlobPerson>& getAllBlobPersons();

	// GET OUTIMAGE //
	cv::Mat& getOutImage();

};

// CLASS TO STORE DETECTED BLOB PERSONS AND TO RETRIEVE FEATURES LIKE 3D HEAD, HANDS AND POSSIBLE FEET POSITIONS // 
class BlobPerson {

private:
	// CONTOUR //
	std::vector <cv::Point> contour; // each contour is a vector of points

	// AREA OF CONTOUR //
	int area;

	// SIMPLIFIED CURVE, the smaller epsilon the closer to original contour //
	std::vector<cv::Point> approxCurve;

	// BOUNDING BOX and points //
	cv::Rect boundRect;
	cv::Point boundingBoxOrigin;
	cv::Point boundingBoxDiagonal;

	// MIN ENCLOSING CIRCLE CENTER //
	cv::Point2f center;
	// MIN ENCLOSING CIRCLE RADIUS //
	float radius;

	// ROTATED BOUNDING BOX //
	cv::RotatedRect rotatedBoundRect;
	cv::Mat boxPoints2f, boxPointsCov;

	// ROTATED ELLIPSE //
	cv::RotatedRect rotatedEllipse;

	//ORIENTATION//
	float angle;

	// ASPECT RATIO ROTATED //
	float aspect_ratio_rotated;

	// AREA OF ROTATED BOUDING BOX //
	float rect_rotated_area;

	// ROTATED EXTEND = ratio of width to height of bounding rect of the object //
	float extend;

	// STRAIGHT LINE APPROXIMATE DATA //
	cv::Vec4f lineData;

	// CONVEX HULL //
	std::vector<cv::Point> hull;

	// AREA OF CONVEX HULL //
	int hull_area;

	// SOLIDITY =  ratio of contour area to its convex hull area.
	float solidity;

	// EXTREME POINTS //
	std::vector<cv::Point3i> extremePoints;

	// EXTREME POINTS SORTED BY HEIGHT //
	std::vector<cv::Point3i> extremePointsByHeight;

	// CENTER OF EXTREME POINTS //
	cv::Point extCenter;

	// CONVEXITY DEFECTS //
	std::vector<cv::Vec4i> defects; // vector of all convexity defects: index start point of approxCurve , index end point of approxCurve , defect depth (distance between the farthest point and the convex hull)

	// HEAD / UPPERBODY CENTER //
	cv::Point3i headCenter;

	// COLORS //
	cv::Scalar red = cv::Scalar(0, 0, 255);
	cv::Scalar green = cv::Scalar(0, 255, 0);
	cv::Scalar blue = cv::Scalar(255, 0, 0);
	cv::Scalar cyan = cv::Scalar(255, 255, 0);
	cv::Scalar yellow = cv::Scalar(0, 255, 255);
	cv::Scalar pink = cv::Scalar(255, 0, 255);

public:
	//INITIALIZE EMPTY BLOB PERSON //
	BlobPerson();

	// INITIALIZE BLOB PERSON WITH CONTOUR //
	BlobPerson(std::vector <cv::Point> contour);

	// GENERATE ALL FEATURES OF BLOB PERSON FROM CONTOUR //
	void calculate();

	//-----------------------------------------------------------------
	// GET FUNCTIONS --------------------------------------------------
	//-----------------------------------------------------------------

	// GET CONTOUR //
	std::vector <cv::Point>& getContour();

	// GET AREA OF CONTOUR //
	int& getArea();

	// GET SIMPLIFIED CURVE, the smaller epsilon the closer to original contour //
	std::vector<cv::Point>& getApproxCurve();

	// GET BOUNDING BOX and points //
	cv::Rect& getBoundRect();
	cv::Point& getBoundingBoxOrigin();
	cv::Point& getBoundingBoxDiagonal();

	// GET MIN ENCLOSING CIRCLE CENTER //
	cv::Point2f& getCenter();

	// MIN ENCLOSING CIRCLE RADIUS //
	float& getRadius();

	// GET ROTATED BOUNDING BOX //
	cv::RotatedRect& getRotatedBoundRect();
	cv::Mat& getBoxPoints2f();
	cv::Mat& getBoxPointsCov();

	// GET ROTATED ELLIPSE //
	cv::RotatedRect& getRotatedEllipse();

	// GET ORIENTATION//
	float& getAngle();

	// GET ASPECT RATIO ROTATED //
	float& getAspect_ratio_rotated();

	// GET AREA OF ROTATED BOUDING BOX //
	float& getRect_rotated_area();

	// GET ROTATED EXTEND = ratio of width to height of bounding rect of the object //
	float& getExtend();

	// GET STRAIGHT LINE APPROXIMATE DATA //
	cv::Vec4f& getLineData();

	// GET CONVEX HULL //
	std::vector<cv::Point>& getHull();

	// GET AREA OF CONVEX HULL //
	int& getHull_area();

	// GET SOLIDITY =  ratio of contour area to its convex hull area.
	float& getSolidity();

	// GET EXTREME POINTS //
	std::vector<cv::Point3i>& getExtremePoints();

	// EXTREME POINTS SORTED BY HEIGHT //
	std::vector<cv::Point3i>& getExtremePointsByHeight();

	// GET EXTREME POINTS Individual //
	cv::Point3i& getHighestExtremePoint1();
	cv::Point3i& getHighestExtremePoint2();
	cv::Point3i& getHighestExtremePoint3();
	cv::Point3i& getHighestExtremePoint4();
	

	// GET CENTER OF EXTREME POINTS //
	cv::Point& getExtCenter();

	// GET HEAD CENTER //
	cv::Point3i& getHeadCenter();

	// GET CONVEXITY DEFECTS //
	std::vector<cv::Vec4i>& getDefects(); // vector of all convexity defects: index start point of approxCurve , index end point of approxCurve , defect depth (distance between the farthest point and the convex hull)

	//-----------------------------------------------------------------
	// DISPLAY FUNCTIONS ----------------------------------------------
	//-----------------------------------------------------------------

	// DISPLAY ALL DETECTED INFORMATION //
	void displayAll(cv::Mat &outImage);

	// DISPLAY CONTOUR //
	void displayContour(cv::Mat &outImage);

	// DISPLAY SIMPLIFIED CURVE //
	void displaySimplifiedContour(cv::Mat &outImage);

	// DISPLAY BOUNDING BOX and points  //
	void displayBoundingBox(cv::Mat &outImage);

	//DISPLAY MIN ENCLOSING CIRCLE //
	void displayMinEnclosingCircle(cv::Mat &outImage);

	//DISPLAY MIN ENCLOSING CENTER //
	void displayMinEnclosingCenter(cv::Mat &outImage);

	//DISPLAY ROTATED BOUNDING BOX //
	void displayRotatedBoundingBox(cv::Mat &outImage);

	//DISPLAY ROTATED ELLIPSE //
	void displayRotatedEllipse(cv::Mat &outImage);

	// DISPLAY STRAIGHT LINE APPROXIMATE DATA //
	void displayStraightLine(cv::Mat &outImage);

	// DISPLAY CONVEX HULL //
	void displayConvexHull(cv::Mat &outImage);

	// DISPLAY EXTREME POINTS //
	void displayExtremePoints(cv::Mat &outImage);

	// DISPLAY CENTER OF EXTREME POINTS //
	void displayExtremePointsCenter(cv::Mat &outImage);

	// DISPLAY EXTREME POINTS HEIGHT AS TEXT//
	void displayExtremePointHeightsValues(cv::Mat &outImage);

	// DISPLAY  HEAD / UPPERBODY CENTER HEIGHT AS TEXT//
	void displayHeadHeightValue(cv::Mat &outImage);

	// DISPLAY CONVEXITY DEFECTS //
	void displayConvexityDefects(cv::Mat &outImage);

	// DISPLAY HEAD / UPPERBODY CENTER //
	void displayHeadCenter(cv::Mat &outImage);

	// DISPLAY ORIENTATION//
	void displayAngleValue(cv::Mat &outImage);

	// DISPLAY ASPECT RATIO ROTATED //
	void displayAspectRatioRotatedValue(cv::Mat &outImage);

	// DISPLAY ROTATED EXTEND = ratio of width to height of bounding rect of the object //
	void displayExtendValue(cv::Mat &outImage);

	// SOLIDITY =  ratio of contour area to its convex hull area.
	void displaySolidityValue(cv::Mat &outImage);

	//-----------------------------------------------------------------
	// SET FUNCTIONS --------------------------------------------------
	//-----------------------------------------------------------------

	// SET EXTREMEPOINT HEIGHT AT INDEX //
	void setExtremePointHeight(int _index, int _height);

	// SET HEAD HEIGHT //
	void setHeadPointHeight(int  _height);

	// SET HEAD POINT LOCATION //
	void setHeadPointLocation(cv::Point _location);

};

// DATA CONVERSION //
cv::Mat convertAnyMatTo_CV_8U_Mat(cv::Mat inputMat);

// GET BIGGEST CONTOUR OUT OF VECTOR OF CONTOURS //
std::vector <cv::Point> getBiggestContour(std::vector<std::vector<cv::Point>> contours);

