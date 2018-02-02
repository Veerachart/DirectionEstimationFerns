/*
 * FisheyeFerns.cpp
 *
 *  Created on: Jan 30, 2018
 *      Author: veerachart
 */

#include "ferns.h"
#include "fern_based_classifier.h"
#include <iostream>
#include <string>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <time.h>
using namespace cv;
using namespace std;

string path_annotate = "/home/veerachart/Python-dir/PIROPO_annotated/omni_1A/omni1A_training/";
string path_original = "/home/veerachart/Python-dir/Dataset_PIROPO/omni_1A/omni_1A/omni1A_training/";
char csvname[] = "/home/veerachart/Python-dir/PIROPO_annotated/omni_1A/omni1A_training/omni1A_training.csv";

Mat visualize(Mat orig_img, Mat cropped_enlarged, int result_angle, int result_cat, double dir_angle, int orig_angle, int orig_cat, vector<float>descriptors, vector<float> descriptors_original);
Mat draw_hog(vector<float>& hog_des);

int main(int argc, char ** argv) {
	fern_based_classifier * classifier;
	char classifier_name[] = "classifiers/classifier_acc_400-4";
	classifier = new fern_based_classifier(classifier_name);

	int hog_size = classifier->hog_image_size;
	FisheyeHOGDescriptor hog(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
	HOGDescriptor hog_original(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
	hog.setAngleMatrix(Size(1600,1600));

	ifstream f(csvname);

	/*string sName, sIdx, sPersonId, sCenterX, sCenterY, sWidth, sHeight;
	string name;
	int idx, personId, centerX, centerY, width, height;
	float angle;

	f >> sName >> sIdx >> sPersonId >> sCenterX >> sCenterY >> sWidth >> sHeight;
	cout << sName << " " << sIdx << " " << sPersonId << " " << sCenterX << " " << sCenterY << " " << sWidth << " " << sHeight << endl;
	f >> name >> idx >> personId >> centerX >> centerY >> width >> height;
	cout << name << " " << idx << " " << personId << " " << centerX << " " << centerY << " " << width << " " << height << endl;*/

	string line_s;
	getline(f, line_s);
	int num_samples = 0;
	string file_name;
	int idx, personId, centerX, centerY, width, height;
	double angle;
	vector<float> descriptors;
	vector<float> descriptors_original;
	while(getline(f, line_s)) {
		stringstream linestream(line_s);
		string value;
		num_samples++;
		vector<string> buffer;
		while(getline(linestream, value,','))
			buffer.push_back(value);
		file_name = buffer[0];
		idx = atoi(buffer[1].c_str());
		personId = atoi(buffer[2].c_str());
		centerX = atoi(buffer[3].c_str());
		centerY = atoi(buffer[4].c_str());
		width = atoi(buffer[5].c_str());
		height = atoi(buffer[6].c_str());
		angle = atof(buffer[7].c_str());

		cout << centerX << " " << centerY << " " << width << " " << height << " " << angle <<  endl;

		char cropped_name[50];
		sprintf(cropped_name, "%06d_%02d.jpg", idx, personId);
		Mat original_image = imread(path_original+file_name);
		Mat padded_image;
		copyMakeBorder(original_image,padded_image,100,100,0,0,BORDER_CONSTANT,Scalar(0,0,0));
		Mat cropped_image = imread(path_annotate+string(cropped_name));
		Mat cropped_resize;
		resize(cropped_image, cropped_resize, Size(hog_size, hog_size));
		Mat draw;
		padded_image.copyTo(draw);
		vector<RotatedRect> ROIs;
		ROIs.push_back(RotatedRect(Point2f(centerX,centerY+100), Size2f(width,height), angle));
		Point2f vertices[4];
		ROIs[0].points(vertices);
		for (int i = 0; i < 4; i++)
			line(draw,vertices[i],vertices[(i+1)%4],Scalar(0,255,0),2);
		descriptors.clear();
		hog.compute(padded_image, descriptors, ROIs);
		hog_original.compute(cropped_resize, descriptors_original);
		int output_class, output_angle;
		classifier->recognize_interpolate(descriptors, cropped_resize, output_class, output_angle);
		int output_class_original, output_angle_original;
		classifier->recognize_interpolate(descriptors_original, cropped_resize, output_class_original, output_angle_original);
		Mat view = visualize(draw, cropped_image, output_angle, output_class, angle, output_angle_original, output_class_original, descriptors, descriptors_original);
		namedWindow(file_name);
		moveWindow(file_name,20,20);
		imshow(file_name,view);
		char c = waitKey(0);
		if (c == 27)
			break;
		else if (c == 's')
			imwrite("ferns_sample2.jpg",view);
		destroyAllWindows();
	}
}

Mat visualize(Mat orig_img, Mat cropped_image, int result_angle, int result_cat, double dir_angle, int orig_angle, int orig_cat, vector<float> descriptors, vector<float> descriptors_original) {
	Mat base_img(900,1600,CV_8UC3,Scalar(64,64,64));
	Mat cropped_enlarged;
	resize(cropped_image, cropped_enlarged, Size(200,200));
	Point img_origin(50,50);
	Point cropped_origin(1100,50);
	Point direction_center(1100,500);
	Point direction_center_original(1400,500);
	int length = 100;

	orig_img.copyTo(base_img(Rect(img_origin, orig_img.size())));
	cropped_enlarged.copyTo(base_img(Rect(cropped_origin, cropped_enlarged.size())));
	char size_txt[10];
	sprintf(size_txt, "%d", cropped_image.rows);
	putText(base_img, size_txt,Point(1050,160), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);
	sprintf(size_txt, "%d", cropped_image.cols);
	putText(base_img, size_txt,Point(1180,30), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);
	char angle_txt[50];
	sprintf(angle_txt, "Angle: %3d", result_angle);
	putText(base_img, angle_txt, Point(950,300), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);
	sprintf(angle_txt, "Category: %d", result_cat);
	putText(base_img, angle_txt, Point(950,330), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);
	char angle_txt_original[50];
	sprintf(angle_txt_original, "Angle: %3d", orig_angle);
	putText(base_img, angle_txt_original, Point(1250,300), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);
	sprintf(angle_txt_original, "Category: %d", orig_cat);
	putText(base_img, angle_txt_original, Point(1250,330), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);

	Point direction_end = direction_center+Point(length*cos((dir_angle-90.)*CV_PI/180.), length*sin((dir_angle-90.)*CV_PI/180.));
	Point direction_end_original = direction_center_original+Point(length*cos((dir_angle-90.)*CV_PI/180.), length*sin((dir_angle-90.)*CV_PI/180.));

	line(base_img, direction_center, direction_end, Scalar(255,255,255), 2);
	arrowedLine(base_img, direction_end, direction_end+Point(50*cos((dir_angle+90.-result_angle)*CV_PI/180.),50*sin((dir_angle+90.-result_angle)*CV_PI/180.)), Scalar(0,0,255),2);

	line(base_img, direction_center_original, direction_end_original, Scalar(255,255,255), 2);
	arrowedLine(base_img, direction_end_original, direction_end_original+Point(50*cos((dir_angle+90.-orig_angle)*CV_PI/180.),50*sin((dir_angle+90.-orig_angle)*CV_PI/180.)), Scalar(0,0,255),2);

	Mat hog_draw = draw_hog(descriptors);
	Mat hog_draw_original = draw_hog(descriptors_original);
	hog_draw.copyTo(base_img(Rect(Point(950,700),hog_draw.size())));
	hog_draw_original.copyTo(base_img(Rect(Point(1250,700),hog_draw.size())));

	return base_img;
	//imshow(filename, base_img);
	//waitKey(0);
}

Mat draw_hog(vector<float>& hog_des) {
	Mat hog_mat(120,216,CV_8UC3);
	/*cout << "[";
	for (int i = 0; i < 323; i++)
		cout << hog_des[i] << ",";
	cout << hog_des[323] << "]" << endl;*/
	int index = 0;
	for (int col = 0; col < 3; col++) {
		for (int row = 0; row < 3; row++) {
			for (int cell_col = 0; cell_col < 2; cell_col++) {
				for (int cell_row = 0; cell_row < 2; cell_row++) {
					for (int bin = 0; bin < 9; bin++) {
						//cout << index << " ";
						Point top_left((col*2*9 + cell_col*9 + bin)*4, (row*2*1 + cell_row)*20);
						uchar pixel_value = hog_des[index]*512;
						//unsigned int pixel_value = index%256;
						//cout << pixel_value << ",";
						Mat pad(20,4,CV_8UC3,Scalar(pixel_value, pixel_value, pixel_value));
						pad.copyTo(hog_mat(Rect(top_left, Size(pad.cols,pad.rows))));
						index++;
					}
				}
			}
		}
	}
	//imshow("HOG", hog_mat);
	//waitKey(0);
	return hog_mat;
}
