# TopDownPeopleDetection
People silhouettes are detected as contours, with feature extraction like height at extremepoints/extremeties, height of center, etc. OpenCV based library, using depth images/ frames from a top down mounted depth camera, e.g. Kinect, Realsense etc.

![empathySwarm_PeopleDetectionSetup](https://user-images.githubusercontent.com/30211868/130604467-ea2b6a8b-fb76-444e-aabd-af22419944c6.jpg)


This code has been developed by [Katrin Hochschuh | Hochschuh&Donovan](https://hochschuh-donovan.com) as part of the project [Empathy Swarm](https://hochschuh-donovan.com/portfolio/empathy-swarm/) before and as part of [»The Intelligent Museum«](#the-intelligent-museum) at [ZKM | Hertz-Lab](https://zkm.de/en/about-the-zkm/organization/hertz-lab). 

Copyright (c) 2021 Katrin Hochschuh | Hochschuh&Donovan.

For information on usage and redistribution, and for a DISCLAIMER OF ALL
WARRANTIES, see the file, "LICENSE.txt," in this distribution.

BSD Simplified License.


Description
-----------
In c++:

Create a cv::Mat of depth information, type CV_16UC1, from each frame:
e.g. for Realsense Camera: 
		 cv::Mat depthMatrix = cv::Mat(cv::Size(w, h), CV_16UC1, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
		depthMatrix.mul(depth_scale);

Before the frame loop create an instance of TopDownPeopleDetector:
  TopDownPeopleDetector peopleDetector;

Inside the frame loop call:
		peopleDetector.update(depthMatrix);

Afterwards you can retrieve the binary image that was created for debugging:
		cv::Mat testImage = peopleDetector.getOutImage();

and get a list of all detected BlobPersons:
		std::vector<BlobPerson> allBlobPersons = peopleDetector.getAllBlobPersons();
  
looping over this list, you can retrieve different information from the BlobPerson object and also display them by giving a reference to an image on which should be drawn, e.g.:
  
			//Contour center + radius
			cv::Point contourCenter = blobperson.getCenter();
			blobperson.displayMinEnclosingCenter(testImage);
			int contourRadius = blobperson.getRadius();
			blobperson.displayMinEnclosingCircle(testImage);

			// Point head center
			cv::Point3i headCenter = blobperson.getHeadCenter();
			blobperson.displayHeadCenter(testImage);
			blobperson.displayHeadHeightValue(testImage);

			// Extremepoints sorted by highest 
			std::vector<cv::Point3i> extremePointsSortedByHighest = blobperson.getExtremePointsByHeight();
			blobperson.displayExtremePoints(testImage);
			blobperson.displayExtremePointHeightsValues(testImage);

			//Convexhull
			std::vector<cv::Point> hull = blobperson.getHull();
			blobperson.displayConvexHull(testImage);

			//Contour simplified
			std::vector<cv::Point> contour = blobperson.getApproxCurve();
			blobperson.displaySimplifiedContour(testImage);

			//Display All Features of Detected BlobPerson//
			//blobperson.displayAll(testImage);

  
  There is a helper function to display images that are in the wrong type:
			testImage = convertAnyMatTo_CV_8U_Mat(testImage);
			cv::imshow(window_name_depth, testImage);


Requirements
------------

(Software) dependencies:
* [OpenCV 4.1](https://opencv.org/opencv-4-1/)

Operating systems/platforms:
* Windows
