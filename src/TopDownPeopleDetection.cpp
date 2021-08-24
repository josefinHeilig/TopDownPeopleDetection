#include "TopDownPeopleDetection.h"

//-----------------------------------------------------------------------------------------
// INITIALIZE TOP DOWN PEOPLE DETECTOR //
//-----------------------------------------------------------------------------------------

TopDownPeopleDetector::TopDownPeopleDetector()
{

}
// UPDATE BLOBPERSONLIST FROM DEPTH IMAGE FRAME IN MM INT //
void TopDownPeopleDetector::update(cv::Mat depthImage)
{
	if (!depthImage.empty())
	{

		//mask out the upper and lower boundaries
		int rows = depthImage.rows;
		depthImage(cv::Range(0, yMin), cv::Range::all()) = 0; // using row and column boundaries
		depthImage(cv::Range(rows - yMax, rows), cv::Range::all()) = 0; // using row and column boundaries
		//mask out the left and right boundaries
		int cols = depthImage.cols;
		depthImage(cv::Range::all(), cv::Range(0, xMin)) = 0; // using row and column boundaries
		depthImage(cv::Range::all(), cv::Range(cols - xMax, cols)) = 0; // using row and column boundaries


		outImage = depthImage.clone();
		cv::Mat binaryImage = depthImage.clone();
		/*
		cv::cvtColor(detectionImage, detectionImage, cv::COLOR_BGR2GRAY);
		cv::cvtColor(outImage, outImage, cv::COLOR_GRAY2BGR);
		*/
		// MAKE BINARY IMAGE BASED ON CLIPPING PLANES - white what is in range and black what is not //
		cv::inRange(depthImage, cv::Scalar(clippingMinDepth), cv::Scalar(clippingMaxDepth), binaryImage);



		/*
		int limit = depthImage.rows * depthImage.cols;

		ushort* ptr = reinterpret_cast<ushort*>(depthImage.data);
		for (int i = 0; i < limit; i++, ptr++)
		{
			if (*ptr >= clippingMinDepth && *ptr <= clippingMaxDepth) {
				//*ptr = 65535;
			}
			else {
				*ptr = 0;
			}
		}
		*/
 
		// DISCARD WRONG READINGS FROM DEPTH IMAGE THROUGH MASK WITH BINARY IMAGE //
		/*
		cv::Mat depthImageTest = cv::Mat::zeros(depthImage.size(), depthImage.type());
		cv::Mat mask = binaryImage.clone();
		mask.convertTo(mask, CV_8UC3);
		//cv::bitwise_and(depthImageTest, mask, depthImageTest);
		depthImage.copyTo(depthImageTest, mask);
		depthImage = depthImageTest.clone();

		depthImageTest = convertAnyMatTo_CV_8U_Mat(depthImageTest);
		cv::namedWindow("ClippedDepthImage", cv::WINDOW_FREERATIO);
		cv::imshow("ClippedDepthImage", depthImageTest);
		*/

		// IMAGE PREPROCESSING //
		cv::GaussianBlur(binaryImage, binaryImage, cv::Size(3, 3), 0, 0);

		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(1, 1));
		cv::dilate(binaryImage, binaryImage, element);

		cv::Mat detectionImage = convertAnyMatTo_CV_8U_Mat(binaryImage); //opencv operations can only be done on CV_8U
		cv::Mat detectionImageHeads = detectionImage.clone();

		outImage = detectionImage.clone(); //outImage to display detections in color
		cv::cvtColor(outImage, outImage, cv::COLOR_GRAY2BGR);

		// FIND CONTOURS //
		std::vector< std::vector <cv::Point> > contours; // each contour is a vector of points
		std::vector<cv::Vec4i> hierarchy; // hierarchy of contours in nested shapes
		cv::findContours(detectionImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());

		//-----------
		// IMAGE PREPROCESSING TO GET CENTER POINT OF BODY / HEAD //
		// ERODING AWAY THE SILHOUETTES TO "STICKFIGURES"
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15,15));//20,20
		cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
		cv::dilate(detectionImageHeads, detectionImageHeads, kernel1, cv::Point(-1, -1), 3);
		cv::erode(detectionImageHeads, detectionImageHeads, kernel, cv::Point(-1, -1), 3);
		outImage = detectionImageHeads.clone();
		cv::cvtColor(outImage, outImage, cv::COLOR_GRAY2BGR);

		//--------------

		// EMPTY THE LIST OF DETECTIONS OF LAST FRAME //
		allBlobPersons.clear();

		// MAKE SURE CONTOURS WERE DETECTED //
		if (!contours.empty()) {

			//----------------------------------------------------------------------------
			// DETECT BLOB PERSONS, EVEN IF TOUCHING AND SAVE TO LIST OF ALL BLOB PERSONS
			//----------------------------------------------------------------------------
			
			// GO THROUGH ALL DETECTED CONTOURS OF OUTER LEVEL//
			for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {

				int area = (int)cv::contourArea(contours[idx]);

				// DISCARD CONTOURS THAT ARE TO SMALL //
				if (area > thresholdContourArea)
				{
					// DISPLAY CONTOURS //
					//cv::Scalar green = cv::Scalar(0, 255, 0);
					//drawContours(outImage, contours, idx, green, 5, cv::LINE_AA);
					//std::cout << contours.size() << std::endl;

					// GET REGION OF INTEREST //
					// simplified curve, the smaller epsilon the closer to original contour //
					std::vector<cv::Point> approxCurve;
					double epsilon = 0.007*arcLength(contours[idx], true);
					cv::approxPolyDP(contours[idx], approxCurve, epsilon, true);
					if (approxCurve.empty())
					{
						//epsilon = 0.002*arcLength(contours[idx], true);
						//cv::approxPolyDP(contours[idx], approxCurve, epsilon, true);
						approxCurve = contours[idx];
					}

					if (!approxCurve.empty())
					{
						//std::cout << approxCurve.size() << std::endl;
						// USE SIMPLIFIED CURVE TO GET BOUNDING BOX FOR ROI //
						cv::Rect boundRect = cv::boundingRect(approxCurve);
						cv::Mat roi = detectionImageHeads(boundRect);
						cv::Mat roiOrg = detectionImage(boundRect); // IMAGE BEFORE EROSION


						// FIND CONTOURS OF UPPER BODIES / HEAD //
						std::vector< std::vector <cv::Point> > roi_contours;
						std::vector<cv::Vec4i> roi_hierarchy;
						cv::findContours(roi, roi_contours, roi_hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());

						//cv::namedWindow("ROI" + std::to_string(idx) + "_" + std::to_string(idx), cv::WINDOW_FREERATIO);
						//cv::imshow("ROI" + std::to_string(idx) + "_" + std::to_string(idx), roi);

						if (!roi_contours.empty() )
						{

							// THE NUMBER OF CONTOURS IS CRUCIAL TO DEFINE IF ONE SILHOUTTE IS ACTUALLY SEVERAL PEOPLE MERGED //
							int numberOfHeads = roi_contours.size();

							// ONE HEAD DETECTED //
							if (numberOfHeads == 1)
							{
								// CREATE BLOB PERSON //
								BlobPerson newBlobPerson = BlobPerson(contours[idx]);
								std::vector<cv::Point>biggestContour = getBiggestContour(roi_contours);
								BlobPerson tempBlobPerson = BlobPerson(biggestContour); // to get the center point of the head area from the eroded image
								// MOVE HEAD LOCATION TO POSITION IN ORIGINAL IMAGE //
								cv::Point headLocation = tempBlobPerson.getCenter();
								headLocation.x += boundRect.x;
								headLocation.y += boundRect.y;
								newBlobPerson.setHeadPointLocation(headLocation);

								allBlobPersons.push_back(newBlobPerson);

							}
							// SEVERAL HEADS DETECTED - NEED TO SPLIT SILHOUETTE INTO PARTS //
							else if(numberOfHeads > 1)
							{
								cv::Scalar color(255, 0, 255);
								std::string string_numberOfHeads = "numberOfHeads: " + std::to_string(numberOfHeads);
								cv::putText(outImage, string_numberOfHeads, cv::Point(50, 50), 0, 1.0, color, 2);

								for (int j = 0; j < numberOfHeads; j++)
								{
									int roi_area = (int)cv::contourArea(roi_contours[j]);
									// DISCARD CONTOURS THAT ARE TO SMALL //
									if (roi_area > thresholdContourAreaHead) // these contours are smaller than the whole silhouette
									{
										BlobPerson tempBlobPerson = BlobPerson(roi_contours[j]);
										//tempBlobPerson.displayAll(outImage);
										// CREATE MASK WITH DOUBLE THE RADIUS OF THE UPPER BODY SILHOUETTE TO INCLUDE EXTREMETIES AGAIN //
										cv::Mat mask = cv::Mat::zeros(roiOrg.size(), roiOrg.type());
										cv::Mat roi2 = cv::Mat::zeros(roiOrg.size(), roiOrg.type());
										cv::circle(mask, tempBlobPerson.getCenter(), int(tempBlobPerson.getRadius() * 2), cv::Scalar(255), -1, 8, 0);
										cv::bitwise_and(roiOrg, mask, roi2);
										// FIND CONTOURS ON MASKED IMAGE TO CREATE A NEW BLOB PERSON TO ADD TO THE LIST //
										std::vector< std::vector <cv::Point> > roi2_contours;
										std::vector<cv::Vec4i> roi2_hierarchy;
										cv::findContours(roi2, roi2_contours, roi2_hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());

										if (!roi2_contours.empty())
										{
											// GET BIGGEST CONTOUR //
											std::vector<cv::Point> personContour = getBiggestContour(roi2_contours);
											// MOVE CONTOUR TO POSITION IN ORIGINAL IMAGE //
											for (cv::Point &point : personContour) {
												point.x += boundRect.x;
												point.y += boundRect.y;
											}

											// CREATE BLOB PERSON //
											BlobPerson newBlobPerson = BlobPerson(personContour);

											// MOVE HEAD LOCATION TO POSITION IN ORIGINAL IMAGE //
											cv::Point headLocation = tempBlobPerson.getCenter();
											headLocation.x += boundRect.x;
											headLocation.y += boundRect.y;
											newBlobPerson.setHeadPointLocation(headLocation);

											allBlobPersons.push_back(newBlobPerson);
										}
									}
								}
							}
						}
						//----------------------------------------------------------------------------
						// GO THROUGH LIST OF ALL BLOB PERSONS AND ADD HEIGHTS
						//----------------------------------------------------------------------------
						
						for (BlobPerson &blobperson : allBlobPersons)
						{

							// GET HEIGHT OF EXTREME POINTS //
							int index = 0;
							for (cv::Point3i extremePoint : blobperson.getExtremePoints())
							{
								// POSITION TRANSLATED FROM ORIGINAL IMAGE TO CROPPED IMAGE ROI //
								cv::Point3i roi_extremePoint = extremePoint;
								roi_extremePoint.x -= blobperson.getBoundRect().x;
								roi_extremePoint.y -= blobperson.getBoundRect().y;

								// GET THE DEPTH VALUES IN THE ROI //
								cv::Mat roi_depth = depthImage(blobperson.getBoundRect());
								// GET THE BINARY IMAGE IN THE ROI //
								cv::Mat roi = detectionImage(blobperson.getBoundRect());

								// CREATE MASK TO GET MEAN HEIGHT - only take the part of the silhouette and the parts inside a certain radius
								cv::Mat maskCombined = cv::Mat::zeros(roi_depth.size(), roi.type());

								// CIRCULAR MASK //
								cv::Mat maskCircle = cv::Mat::zeros(roi_depth.size(), roi.type());
								int radius = 20;
								cv::circle(maskCircle, cv::Point(roi_extremePoint.x, roi_extremePoint.y), radius, cv::Scalar(255), -1, 8, 0);
								cv::bitwise_and(roi, maskCircle, maskCombined);

								// CALCULATE MEAN OF ALL DEPTH VALUES THAT ARE MASKED //
								cv::Scalar tempVal = cv::mean(roi_depth, { maskCombined });
								int mean_height = (int)tempVal.val[0];
								mean_height = realDepth - mean_height; // to translate depth from camera to height from floor
								blobperson.setExtremePointHeight(index, mean_height);
								index++;

								//cv::namedWindow("Cropped ROI" + std::to_string(index) + "_" + std::to_string(index), cv::WINDOW_FREERATIO);
								//cv::imshow("Cropped ROI" + std::to_string(index) + "_" + std::to_string(index), maskCombined);
							}

							// GET HEIGHEST POINT //
							// GET THE DEPTH VALUES IN THE ROI //
							cv::Mat roi_depth = depthImage(blobperson.getBoundRect());

							// GET THE BINARY IMAGE IN THE ROI //
							cv::Mat roi = detectionImage(blobperson.getBoundRect());

							// CREATE MASK TO GET MEAN HEIGHT - only take the part of the silhouette and the parts inside a certain radius
							cv::Mat maskCombined = cv::Mat::zeros(roi_depth.size(), roi.type());

							// CIRCULAR MASK //
							cv::Mat maskCircle = cv::Mat::zeros(roi_depth.size(), roi.type());

							cv::Point3i posTemp = blobperson.getHeadCenter();
							int radius = blobperson.getRadius();// = 20;
							cv::Point pos = cv::Point(posTemp.x, posTemp.y);
							cv::Point pos_roi = pos;
							pos_roi.x -= blobperson.getBoundRect().x;
							pos_roi.y -= blobperson.getBoundRect().y;

							cv::circle(maskCircle, pos_roi, radius, cv::Scalar(255), -1, 8, 0);
							cv::bitwise_and(roi, maskCircle, maskCombined);

							// CALCULATE MEAN OF ALL DEPTH VALUES THAT ARE MASKED //
							cv::Scalar tempVal = cv::mean(roi_depth, { maskCombined });
							int mean_height = (int)tempVal.val[0];
							mean_height = realDepth - mean_height; // to translate depth from camera to height from floor
							blobperson.setHeadPointHeight(mean_height);
							

							// DOES NOT WORK WELL ENOUGH, TOO MANY HIGH VALUES OF SAME VALUES, SO JITTERY //
							/*
							cv::Point highestLocation;
							double highestValue;
							getHeighestValueLocationInMat(roi_depth, highestLocation, highestValue);
							int height = realDepth - (int)highestValue; // to translate depth from camera to height from floor
							highestLocation.x += blobperson.getBoundRect().x;
							highestLocation.y += blobperson.getBoundRect().y;
							blobperson.setHeadPointLocation(highestLocation);
							blobperson.setHeadPointHeight(height);*/
							
							// SORT EXTREMEPOINTS BY HEIGHT //
							struct pointSort
							{
								inline bool operator() (const cv::Point3i& struct1, const cv::Point3i& struct2)
								{
									if (struct1.z == struct2.z)
										return (struct1.z > struct2.z);
									else
										return struct1.z > struct2.z;
								}
							};
							// CREATE COPY OF EXTREME POINTS IN EXTREMEPINTS SORTED BY HEIGHT //
							for (cv::Point3i extremePoint : blobperson.getExtremePoints())
							{
								blobperson.getExtremePointsByHeight().push_back(extremePoint);
								
							}
							// using pointSort as compare
							std::sort(blobperson.getExtremePointsByHeight().begin(), blobperson.getExtremePointsByHeight().end(),pointSort());

							/*for (cv::Point3i extremePoint : blobperson.getExtremePointsByHeight())
							{
								std::cout << extremePoint.z << std::endl;
							}
							std::cout << "---------" << std::endl;
							for (cv::Point3i extremePoint : blobperson.getExtremePoints())
							{
								std::cout << extremePoint.z << std::endl;
							}
							std::cout << "---------" << std::endl;
							std::cout << "---------" << std::endl;*/
						}

						

						//----------------------------------------------------------------------------
						// DISPLAY ALL DETECTED FEATURES OF ALL BLOBPERSONS//
						//----------------------------------------------------------------------------
						/*for (BlobPerson &blobperson : allBlobPersons)
						{
							blobperson.displayAll(outImage);
						}*/
					}
				}
			}
		}
	}
}

// GET BLOBPERSON LIST //
std::vector<BlobPerson>& TopDownPeopleDetector::getAllBlobPersons()
{
	return allBlobPersons;
}

// GET OUTIMAGE //
cv::Mat& TopDownPeopleDetector::getOutImage()
{
	//----------------------------------------------------------------------------
	// DISPLAY ALL DETECTED FEATURES OF ALL BLOBPERSONS//
	//----------------------------------------------------------------------------
	/*for (BlobPerson &blobperson : allBlobPersons)
	{
		blobperson.displayAll(outImage);
	}*/
	return outImage;
}


//-----------------------------------------------------------------------------------------
// INITIALIZE BLOB PERSON //
//-----------------------------------------------------------------------------------------

BlobPerson::BlobPerson()
{

}

// INITIALIZE BLOB PERSON WITH CONTOUR //
BlobPerson::BlobPerson(std::vector <cv::Point> _contour)
{
	contour = _contour;
	calculate();
}

void BlobPerson::calculate()
{
	// AREA //
	area = (int)cv::contourArea(contour);

	// SIMPLIFIED CURVE //
	double epsilon = 0.007*arcLength(contour, true); //the smaller epsilon the closer to original contour , 0.002 org
	cv::approxPolyDP(contour, approxCurve, epsilon, true);
	if (approxCurve.empty()) {
		approxCurve = contour;
	}

	//BOUNDING BOX and points //
	boundRect = cv::boundingRect(approxCurve);
	boundingBoxOrigin = boundRect.tl();
	boundingBoxDiagonal = boundRect.br();

	// MIN ENCLOSING CIRCLE //
	cv::minEnclosingCircle(approxCurve, center, radius);

	// ROTATED BOUNDING BOX //
	rotatedBoundRect = minAreaRect(approxCurve);
	boxPoints(rotatedBoundRect, boxPoints2f);
	boxPoints2f.assignTo(boxPointsCov, CV_32S);

	// ROTATED ELLIPSE //
	rotatedEllipse = fitEllipse(approxCurve);

	//ORIENTATION//
	angle = rotatedEllipse.angle;

	// ASPECT RATIO ROTATED //
	aspect_ratio_rotated = rotatedEllipse.size.width / rotatedEllipse.size.height;

	// ROTATED EXTEND = ratio of width to height of bounding rect of the object //
	rect_rotated_area = rotatedEllipse.size.width * rotatedEllipse.size.height;
	extend = area / rect_rotated_area;

	// STRAIGHT LINE APPROXIMATE DATA //
	fitLine(approxCurve, lineData, cv::DIST_L2, 0, 0.01, 0.01);

	// CONVEX HULL //
	cv::convexHull(contour, hull);

	//SOLIDITY =  ratio of contour area to its convex hull area.
	hull_area = (int)cv::contourArea(hull);
	solidity = float(area) / hull_area;

	//EXTREME POINTS //
	// LEFT //
	cv::Point extLeft = *std::min_element(contour.begin(), contour.end(),
		[](cv::Point &lhs, const cv::Point &rhs) {
		return lhs.x < rhs.x;
	});
	extremePoints.push_back(cv::Point3i(extLeft.x, extLeft.y, 0));
	// RIGHT //
	cv::Point extRight = *std::max_element(contour.begin(), contour.end(),
		[](const cv::Point &lhs, const cv::Point &rhs) {
		return lhs.x < rhs.x;
	});
	extremePoints.push_back(cv::Point3i(extRight.x, extRight.y, 0));
	// TOP //
	cv::Point extTop = *std::min_element(contour.begin(), contour.end(),
		[](const cv::Point &lhs, const cv::Point &rhs) {
		return lhs.y < rhs.y;
	});
	extremePoints.push_back(cv::Point3i(extTop.x, extTop.y, 0));
	// BOTTOM //
	cv::Point extBot = *std::max_element(contour.begin(), contour.end(),
		[](const cv::Point &lhs, const cv::Point &rhs) {
		return lhs.y < rhs.y;
	});
	extremePoints.push_back(cv::Point3i(extBot.x, extBot.y, 0));

	// CENTER OF EXTREME POINTS //
	extCenter = (extLeft + extRight + extTop + extBot) / 4;

	// CONVEXITY DEFECTS //
	std::vector<int> hull2;
	convexHull(approxCurve, hull2, false, false);
	// vector of all convexity defects: index start point of approxCurve , index end point of approxCurve , defect depth (distance between the farthest point and the convex hull)
	convexityDefects(approxCurve, hull2, defects);
}


//-----------------------------------------------------------------
// GET FUNCTIONS --------------------------------------------------
//-----------------------------------------------------------------

// GET CONTOUR //
std::vector <cv::Point>& BlobPerson::getContour()
{
	return contour;
}

// GET AREA OF CONTOUR //
int& BlobPerson::getArea()
{
	return area;
}

// GET SIMPLIFIED CURVE, the smaller epsilon the closer to original contour //
std::vector<cv::Point>& BlobPerson::getApproxCurve()
{
	return approxCurve;
}

// GET BOUNDING BOX and points //
cv::Rect& BlobPerson::getBoundRect()
{
	return boundRect;
}

cv::Point& BlobPerson::getBoundingBoxOrigin()
{
	return boundingBoxOrigin;
}

cv::Point& BlobPerson::getBoundingBoxDiagonal()
{
	return boundingBoxDiagonal;
}

// GET MIN ENCLOSING CIRCLE CENTER //
cv::Point2f& BlobPerson::getCenter()
{
	return center;
}

// MIN ENCLOSING CIRCLE RADIUS //
float& BlobPerson::getRadius()
{
	return radius;
}

// GET ROTATED BOUNDING BOX //
cv::RotatedRect& BlobPerson::getRotatedBoundRect()
{
	return rotatedBoundRect;
}

cv::Mat& BlobPerson::getBoxPoints2f()
{
	return boxPoints2f;
}

cv::Mat& BlobPerson::getBoxPointsCov()
{
	return boxPointsCov;
}

// GET ROTATED ELLIPSE //
cv::RotatedRect& BlobPerson::getRotatedEllipse()
{
	return rotatedEllipse;
}

// GET ORIENTATION//
float& BlobPerson::getAngle()
{
	return angle;
}

// GET ASPECT RATIO ROTATED //
float& BlobPerson::getAspect_ratio_rotated()
{
	return aspect_ratio_rotated;
}

// GET AREA OF ROTATED BOUDING BOX //
float& BlobPerson::getRect_rotated_area()
{
	return rect_rotated_area;
}

// GET ROTATED EXTEND = ratio of width to height of bounding rect of the object //
float& BlobPerson::getExtend()
{
	return extend;
}

// GET STRAIGHT LINE APPROXIMATE DATA //
cv::Vec4f& BlobPerson::getLineData()
{
	return lineData;
}

// GET CONVEX HULL //
std::vector<cv::Point>& BlobPerson::getHull()
{
	return hull;
}

// GET AREA OF CONVEX HULL //
int& BlobPerson::getHull_area()
{
	return hull_area;
}

// GET SOLIDITY =  ratio of contour area to its convex hull area.
float& BlobPerson::getSolidity()
{
	return solidity;
}

// GET EXTREME POINTS //
std::vector<cv::Point3i>& BlobPerson::getExtremePoints()
{
	return extremePoints;
}

// EXTREME POINTS SORTED BY HEIGHT //
std::vector<cv::Point3i>& BlobPerson::getExtremePointsByHeight()
{
	return extremePointsByHeight;
}


// GET EXTREME POINTS Individual //
cv::Point3i& BlobPerson::getHighestExtremePoint1()
{
	return extremePointsByHeight.at(0);
}
cv::Point3i& BlobPerson::getHighestExtremePoint2()
{
	return extremePointsByHeight.at(1);
}
cv::Point3i& BlobPerson::getHighestExtremePoint3()
{
	return extremePointsByHeight.at(3);
}
cv::Point3i& BlobPerson::getHighestExtremePoint4()
{
	return extremePointsByHeight.at(4);
}


// GET CENTER OF EXTREME POINTS //
cv::Point& BlobPerson::getExtCenter()
{
	return extCenter;
}

// GET HEAD CENTER //
cv::Point3i& BlobPerson::getHeadCenter()
{
	return headCenter;
}

// GET CONVEXITY DEFECTS //
// vector of all convexity defects: index start point of approxCurve , index end point of approxCurve , defect depth (distance between the farthest point and the convex hull)
std::vector<cv::Vec4i>& BlobPerson::getDefects()
{
	return defects;
}

//-----------------------------------------------------------------
// DISPLAY FUNCTIONS ----------------------------------------------
//-----------------------------------------------------------------

// DISPLAY CONTOUR //
void BlobPerson::displayContour(cv::Mat &outImage)
{
	std::vector<std::vector<cv::Point>> poolCurves;
	poolCurves.push_back(contour);
	drawContours(outImage, poolCurves, 0, green, 1, cv::LINE_AA);
}
// DISPLAY SIMPLIFIED CURVE //
void BlobPerson::displaySimplifiedContour(cv::Mat &outImage)
{
	std::vector<std::vector<cv::Point>> poolApproxCurve;
	poolApproxCurve.push_back(approxCurve);
	drawContours(outImage, poolApproxCurve, 0, red, 1, cv::LINE_AA);
}
// DISPLAY BOUNDING BOX and points  //
void BlobPerson::displayBoundingBox(cv::Mat &outImage)
{
	cv::rectangle(outImage, boundingBoxOrigin, boundingBoxDiagonal, green, 1);
	cv::circle(outImage, boundingBoxOrigin, 5, green, -1);
	cv::circle(outImage, boundingBoxDiagonal, 5, green, -1);
}

// DISPLAY MIN ENCLOSING CIRCLE //
void BlobPerson::displayMinEnclosingCircle(cv::Mat &outImage)
{
	cv::circle(outImage, center, (int)radius, green, 1);
}

//DISPLAY MIN ENCLOSING CENTER //
void BlobPerson::displayMinEnclosingCenter(cv::Mat &outImage)
{
	cv::circle(outImage, center, 5, green, -1);
}

// DISPLAY ROTATED BOUNDING BOX //
void BlobPerson::displayRotatedBoundingBox(cv::Mat &outImage)
{
	polylines(outImage, boxPointsCov, true, blue, 1);
}

// DISPLAY ROTATED ELLIPSE //
void BlobPerson::displayRotatedEllipse(cv::Mat &outImage)
{
	ellipse(outImage, rotatedEllipse, blue, 1);
}

// DISPLAY STRAIGHT LINE APPROXIMATE DATA //
void BlobPerson::displayStraightLine(cv::Mat &outImage)
{
	int lefty = (-lineData[2] * lineData[1] / lineData[0]) + lineData[3];
	int righty = ((outImage.cols - lineData[2])*lineData[1] / lineData[0]) + lineData[3];
	line(outImage, cv::Point(outImage.cols - 1, righty), cv::Point(0, lefty), green, 1);

}

// DISPLAY CONVEX HULL //
void BlobPerson::displayConvexHull(cv::Mat &outImage)
{
	std::vector<std::vector<cv::Point>> poolCurves;
	poolCurves.push_back(hull);
	drawContours(outImage, poolCurves, 0, pink, 2, cv::LINE_AA);
}

// DISPLAY EXTREME POINTS //
void BlobPerson::displayExtremePoints(cv::Mat &outImage)
{
	for (int i = 0; i < extremePoints.size(); i++)
	{
		cv::circle(outImage, cv::Point(extremePoints.at(i).x, extremePoints.at(i).y), 8, yellow, -1);
	}
}

// DISPLAY CENTER OF EXTREME POINTS //
void BlobPerson::displayExtremePointsCenter(cv::Mat &outImage)
{
	cv::circle(outImage, extCenter, 8, yellow, -1);
}

// DISPLAY CONVEXITY DEFECTS //
void BlobPerson::displayConvexityDefects(cv::Mat &outImage)
{
	cv::Point ptStartFirst = cv::Point(-1, -1);
	cv::Point ptEndLast = cv::Point(-1, -1);

	for (int i = 0; i < defects.size(); i++) {
		cv::Vec4i v = defects[i];
		float defects_depth = v[3]; // distance between the farthest point and the convex hull
		if (defects_depth > 10) {
			int startidx = v[0];
			cv::Point ptStart(approxCurve[startidx]);
			int endidx = v[1];
			cv::Point ptEnd(approxCurve[endidx]);
			int faridx = v[2];
			cv::Point ptFar(approxCurve[faridx]);

			line(outImage, ptStart, ptEnd, cyan, 1);
			circle(outImage, ptFar, 5, cyan, -1);

			line(outImage, ptStart, ptFar, cyan, 3);
			line(outImage, ptFar, ptEnd, cyan, 3);


			/*if (ptEndLast.x != -1) line(outImage, ptEndLast, ptStart, cyan, 3);
			if (i == 0) ptStartFirst = ptStart;
			if (ptStartFirst.x != -1 && i == defects.size()-1) line(outImage, ptEnd, ptStartFirst, cyan, 3);
			ptEndLast = ptEnd;*/
		}
	}
}

// DISPLAY HEAD / UPPERBODY CENTER //
void BlobPerson::displayHeadCenter(cv::Mat &outImage)
{
	cv::circle(outImage, cv::Point(headCenter.x, headCenter.y), 8, pink, -1);
}

// DISPLAY EXTREME POINTS HEIGHT AS TEXT//
void BlobPerson::displayExtremePointHeightsValues(cv::Mat &outImage)
{
	for (int i = 0; i < extremePoints.size(); i++)
	{
		std::string stringValue = "P " + std::to_string(extremePoints.at(i).z);
		cv::Point pos = cv::Point(extremePoints.at(i).x - 20, extremePoints.at(i).y - 20);
		cv::putText(outImage, stringValue, pos, 0, 0.5, red, 2);
	}
}

// DISPLAY  HEAD / UPPERBODY CENTER HEIGHT AS TEXT//
void BlobPerson::displayHeadHeightValue(cv::Mat &outImage)
{
	std::string stringValue = "H " + std::to_string(headCenter.z);
	cv::Point pos = cv::Point(headCenter.x - 20, headCenter.y - 20);
	cv::putText(outImage, stringValue, pos, 0, 0.5, pink, 2);

}
// DISPLAY ORIENTATION//
void BlobPerson::displayAngleValue(cv::Mat &outImage)
{
	std::string stringValue = "angle: " + std::to_string(angle);
	cv::Point pos = cv::Point(center.x, center.y - 20);
	cv::putText(outImage, stringValue, pos, 0, 0.5, red, 2);
}

// DISPLAY ASPECT RATIO ROTATED //
void BlobPerson::displayAspectRatioRotatedValue(cv::Mat &outImage)
{
	std::string stringValue = "aspect_ratio_rotated: " + std::to_string(aspect_ratio_rotated);
	cv::Point pos = cv::Point(center.x, center.y - 20);
	cv::putText(outImage, stringValue, pos, 0, 0.5, red, 2);
}

// DISPLAY ROTATED EXTEND = ratio of width to height of bounding rect of the object //
void BlobPerson::displayExtendValue(cv::Mat &outImage)
{
	std::string stringValue = "extend: " + std::to_string(extend);
	cv::Point pos = cv::Point(center.x, center.y - 20);
	cv::putText(outImage, stringValue, pos, 0, 0.5, red, 2);
}

// SOLIDITY =  ratio of contour area to its convex hull area.
void BlobPerson::displaySolidityValue(cv::Mat &outImage)
{
	std::string stringValue = "solidity: " + std::to_string(solidity);
	cv::Point pos = cv::Point(center.x, center.y - 20);
	cv::putText(outImage, stringValue, pos, 0, 0.5, red, 2);
}

// DISPLAY ALL //
void BlobPerson::displayAll(cv::Mat &outImage)
{
	// DISPLAY CONTOUR //
	displayContour(outImage);

	// DISPLAY SIMPLIFIED CURVE //
	displaySimplifiedContour(outImage);

	// DISPLAY BOUNDING BOX and points  //
	displayBoundingBox(outImage);

	//DISPLAY MIN ENCLOSING CIRCLE //
	displayMinEnclosingCircle(outImage);

	//DISPLAY MIN ENCLOSING CENTER //
	displayMinEnclosingCenter(outImage);

	//DISPLAY ROTATED BOUNDING BOX //
	displayRotatedBoundingBox(outImage);

	//DISPLAY ROTATED ELLIPSE //
	displayRotatedEllipse(outImage);

	// DISPLAY STRAIGHT LINE APPROXIMATE DATA //
	displayStraightLine(outImage);

	// DISPLAY CONVEX HULL //
	displayConvexHull(outImage);

	// DISPLAY EXTREME POINTS //
	displayExtremePoints(outImage);

	// DISPLAY CENTER OF EXTREME POINTS //
	displayExtremePointsCenter(outImage);

	// DISPLAY CONVEXITY DEFECTS //
	displayConvexityDefects(outImage);

	// DISPLAY HEAD / UPPERBODY CENTER //
	displayHeadCenter(outImage);

	// DISPLAY EXTREME POINTS HEIGTHS AS TEXT //
	displayExtremePointHeightsValues(outImage);

	// DISPLAY  HEAD / UPPERBODY CENTER HEIGHT AS TEXT//
	displayHeadHeightValue(outImage);

}

//-----------------------------------------------------------------
// SET FUNCTIONS --------------------------------------------------
//-----------------------------------------------------------------

// SET EXTREMEPOINT HEIGHT AT INDEX //
void BlobPerson::setExtremePointHeight(int _index, int  _height)
{
	extremePoints.at(_index).z = _height;
}

// SET HEAD HEIGHT //
void BlobPerson::setHeadPointHeight(int  _height)
{
	headCenter.z = _height;
}
// SET HEAD POINT LOCATION //
void BlobPerson::setHeadPointLocation(cv::Point _location)
{
	headCenter.x = _location.x;
	headCenter.y = _location.y;
}

// DATA CONVERSION //
cv::Mat convertAnyMatTo_CV_8U_Mat(cv::Mat inputMat)
{
	double min;
	double max;
	cv::minMaxIdx(inputMat, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(inputMat, adjMap, 255 / max);
	return adjMap;
}

// GET BIGGEST CONTOUR OUT OF VECTOR OF CONTOURS //
std::vector <cv::Point> getBiggestContour(std::vector<std::vector<cv::Point>> contours)
{
	auto biggestContour = *std::max_element(contours.begin(), contours.end(), [](std::vector<cv::Point> const& lhs, std::vector<cv::Point> const& rhs)
	{
		return cv::contourArea(lhs, false) < cv::contourArea(rhs, false);
	});
	return biggestContour;
}

//GET LOCATION AND VALUE OF HEIGHEST VALUE IN MAT //
void TopDownPeopleDetector::getHeighestValueLocationInMat(cv::Mat& m, cv::Point& maxLoc, double& maxVal)
{
	//double minVal;
	//double maxVal;
	//cv::Point minLoc;
	//cv::Point maxLoc;
	//cv::minMaxLoc(m, &minVal, &maxVal, &minLoc, &maxLoc);
	//cv::minMaxLoc(m, &maxVal, &minVal, &maxLoc, &minLoc);


	// GET LOWEST VALUE IN RANGE OF CLIPPING MIN AND MAX //

	int limit = m.rows * m.cols;

	int smallesVal = 65535;
	int smallesValIndex = 0;

	int* ptr = reinterpret_cast<int*>(m.data);
	for (int i = 0; i < limit; i++, ptr++)
	{
		if (*ptr >= clippingMinDepth && *ptr <= clippingMaxDepth && *ptr < smallesVal) {
			smallesVal = *ptr;
			smallesValIndex = i;
			std::cout << smallesVal << std::endl;
		}

	}
	std::cout << "------------------" << std::endl;
	maxVal = smallesVal;
	maxLoc.x = smallesValIndex % m.cols;
	maxLoc.y = ceil (smallesValIndex / m.cols);

}