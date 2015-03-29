#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String face_cascade_name = "haarcascade_frontalface_default.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";

int main(int argc, char** argv)
{
	
	// Get filename from input and prepare image
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}
	Mat image= imread(argv[1], IMREAD_COLOR); // Read the file
	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// Prepare data structures and image
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(image, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- 1. Load the cascades and detect faces
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 5, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	// Write cropped results
	for (size_t i = 0; i < faces.size(); i++)
	{
		imwrite("face" + std::to_string(i) + ".jpg", image(faces[i]));
	}

	return 0;
}