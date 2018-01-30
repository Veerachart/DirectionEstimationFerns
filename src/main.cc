/*
 * main.cc
 *
 *  Created on: Dec 28, 2017
 *      Author: Veerachart Srisamosorn
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

string path = "/home/otalab/dataset/";

int training_data = 0;
const int test_data = 300;		// Samples used for testing
vector<Mat> images;
vector<Mat> test_images;
vector<int> ground_truths;
vector<int> ground_angles;
vector<int> test_ground_truths;
vector<int> test_angles;
vector<string> train_names;
vector<string> test_names;
int count_test = 0;

void ProcessDirectory(string directory, vector<string>& file_list);
void ProcessEntity(struct dirent* entity, vector<string>& file_list);
void ProcessFile(string dir, string file, int img_size);
int myrandom(int i) {return rand()%i;}
void mean_and_sd (double data[], size_t length, double& mean, double& sd);
double max(double data[], size_t length, int& max_idx);
double min(double data[], size_t length, int& min_idx);

int main(int argc, char ** argv) {
	/*fern_based_classifier * classifier;
	char classifier_name[] = "classifier_MAAE";
	classifier = new fern_based_classifier(classifier_name);
	for (int i = 0; i < 3; i++) {
	    for (int j = 0; j < classifier->Ferns->number_of_leaves_per_fern; j++) {
	        for (int k = 0; k < classifier->number_of_classes; k++)
	            cout << classifier->leaves_counters[i*classifier->step1 + j*classifier->step2 + k] << " ";
	        cout << "\t";
	    }
	    cout << endl;
	}
	classifier->save(classifier_name);
	delete classifier;
	return 0;*/
	/*float x = 45, y = 90, z = 135;
	float a = 0.57736;
	float b =91.2116;
	float c = 66.5133;
	double d = x*exp(a) + y*exp(b) + z*exp(c);
	double e = a + b + c;
	cout << a << ". " << b << ". " << c << ". " << d << ". " << e << ". " << d/e << endl;
	printf("%f, %f, %f, %f, %f, %f", a, b, c, d, e, d/e);
	//double estimate = exp(class_score)*class_representations[class_index] + exp(distribution[low])*low_representation + exp(distribution[high])*high_representation;
	//double sum = exp(class_score) + exp(distribution[low]) + exp(distribution[high]);
	return 0;*/

	srand(time(NULL));
	//int64 total_start = getTickCount();
	vector<string> file_list;				// Keep list of file before random
	file_list.reserve(10000);
	images.reserve(2000);
	ground_truths.reserve(2000);
	ground_angles.reserve(2000);
	train_names.reserve(2000);
	test_images.reserve(500);
	test_ground_truths.reserve(500);
	test_angles.reserve(500);
	test_names.reserve(500);
	string directory = "BMVC2009TrainingData";
	ProcessDirectory(directory, file_list);
	int hog_size = 20;
	HOGDescriptor hog(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
	/*int list_ferns[] = {200};//,300,400};
	int list_decisions[] = {4,6};//,8,10};
	size_t length_ferns = sizeof(list_ferns)/sizeof(*list_ferns);
	size_t length_decisions = sizeof(list_decisions)/sizeof(*list_decisions);
	int number_of_pairs = length_ferns*length_decisions;
	int pairs_of_decisions_ferns[number_of_pairs][2];
	for (size_t i = 0; i < length_decisions; i++)
		for (size_t j = 0; j < length_ferns; j++) {
			pairs_of_decisions_ferns[i*length_ferns + j][0] = list_decisions[i];
			pairs_of_decisions_ferns[i*length_ferns + j][1] = list_ferns[j];
		}
	int pairs_of_decisions_ferns[][2] =
	                                      {{3,100}, {4,75},  {5,60},  {6,50},  {7,45},  {8,40},  {9,35},  {10,30},  {12,25},  {16,20},//, {20,15},		// N ~ 300
	                                       {3,200}, {4,150}, {5,120}, {6,100}, {7,90},  {8,80},  {9,70},  {10,60},  {12,50},  {16,40},//, {20,30}, 	// N ~ 600
	                                       {3,300}, {4,225}, {5,180}, {6,150}, {7,135}, {8,120}, {9,105}, {10,90},  {12,75},  {16,60},//, {20,45}, 	// N ~ 900
	                                       {3,400}, {4,300}, {5,240}, {6,200}, {7,180}, {8,160}, {9,140}, {10,120}, {12,100}, {16,80}};//, {20,60}};	// N ~ 1200
	int number_of_pairs = 1;
	int pairs_of_decisions_ferns[][2] = {{3,400}};
	int number_of_random_dataset = 1;
	int number_of_iterations = 1;

	double class_accuracy[number_of_pairs][number_of_iterations*number_of_random_dataset];
	double MAAE[number_of_pairs][number_of_iterations*number_of_random_dataset];
	double computation_time[number_of_pairs][number_of_iterations*number_of_random_dataset];
	for (int it_dataset = 0; it_dataset < number_of_random_dataset; it_dataset++) {
		cout << "Shuffling " << it_dataset << endl;
		// Reshuffling images
		images.clear();
		ground_truths.clear();
		ground_angles.clear();
		train_names.clear();
		test_images.clear();
		test_ground_truths.clear();
		test_angles.clear();
		test_names.clear();
		training_data = 0;
		count_test = 0;
		random_shuffle(file_list.begin(), file_list.end(), myrandom);
		for (unsigned int file_idx = 0; file_idx < file_list.size(); file_idx++) {
			ProcessFile(directory, file_list[file_idx],hog_size);
		}

		for (int pair = 0; pair < number_of_pairs; pair++) {
			// For each decision-fern pair
			int num_decisions = pairs_of_decisions_ferns[pair][0];
			int num_ferns = pairs_of_decisions_ferns[pair][1];

			cout << "> " << num_ferns << " ferns, " << num_decisions << " decisions" << endl;

			for (int iteration = 0; iteration < number_of_iterations; iteration++) {
				if (iteration % 5 == 0)
					cout << "> > Iterations " << iteration << "..." << endl;
				//cout << "> > Iteration " << iteration << "..." << endl;
				// Repeat many iterations as it is random
				fern_based_classifier * classifier;
				vector<vector<float> > descriptors(training_data);

				for (int i = 0; i < training_data; i++) {
					hog.compute(images[i], descriptors[i]);
				}

				classifier = new fern_based_classifier(8,num_ferns,num_decisions,hog_size,false);
				// Training
				classifier->train(descriptors, images, ground_truths, training_data);
				classifier->finalize_training();
				// Testing
				vector<float> test_descriptor;
				int64 start, now, sum_time = 0;
				int error = 0;
				int count_correct_class = 0;
				for (int i = 0; i < test_data; i++) {
					start = getTickCount();
					hog.compute(test_images[i], test_descriptor);
					int output_class, output_angle;
					classifier->recognize_interpolate(test_descriptor, test_images[i], output_class, output_angle);
					now = getTickCount();
					//cout << test_angles[i] << " " << output_old << ", " << output_new << "\t\t" << test_ground_truths[i] << " " << output_class << endl;
					int diff = abs(test_angles[i] - output_angle);
					if (diff > 180)
						error += 360 - diff;
					else
						error += diff;
					if (output_class == test_ground_truths[i])
						count_correct_class++;

					sum_time += now-start;
					//cout << error << " ";
				}
				//cout << endl;

				computation_time[pair][it_dataset*number_of_iterations + iteration] = double(sum_time)/getTickFrequency()/double(test_data);
				class_accuracy[pair][it_dataset*number_of_iterations + iteration] = double(count_correct_class)/double(test_data)*100.0;
				MAAE[pair][it_dataset*number_of_iterations + iteration] = double(error)/double(test_data);
				//cout << computation_time[pair][it_dataset*number_of_iterations + iteration] << ", " << class_accuracy[pair][it_dataset*number_of_iterations + iteration] << ", " << MAAE[pair][it_dataset*number_of_iterations + iteration] << endl;
				delete classifier;
			}
		}
	}
	double mean_computation_time[number_of_pairs];
	double mean_class_accuracy[number_of_pairs];
	double mean_MAAE[number_of_pairs];
	double sd_computation_time[number_of_pairs];
	double sd_class_accuracy[number_of_pairs];
	double sd_MAAE[number_of_pairs];

	for (int pair = 0; pair < number_of_pairs; pair++) {
		//cout << pair << endl;
		mean_and_sd(computation_time[pair], size_t(number_of_iterations*number_of_random_dataset), mean_computation_time[pair], sd_computation_time[pair]);
		mean_and_sd(class_accuracy[pair], size_t(number_of_iterations*number_of_random_dataset), mean_class_accuracy[pair], sd_class_accuracy[pair]);
		mean_and_sd(MAAE[pair], size_t(number_of_iterations*number_of_random_dataset), mean_MAAE[pair], sd_MAAE[pair]);
	}

	char csvname[] = "ferns-decision-newinterpolate.csv";
	FILE * csvfile;
	csvfile = fopen(csvname,"w");

	if (csvfile!=NULL) {
		fprintf(csvfile, "#Ferns,#Decisions,Mean_computation_time,SD_computation_time(ms),Mean_class_accuracy,SD_class_accuracy,Mean_MAAE,SD_MAAE\n");
		for (int pair = 0; pair < number_of_pairs; pair++) {
			int num_decisions = pairs_of_decisions_ferns[pair][0];
			int num_ferns = pairs_of_decisions_ferns[pair][1];

			fprintf(csvfile,"%d,%d,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf\n", num_ferns, num_decisions, mean_computation_time[pair]*1000, sd_computation_time[pair]*1000,
					mean_class_accuracy[pair], sd_class_accuracy[pair], mean_MAAE[pair], sd_MAAE[pair]);
		}
		fclose(csvfile);
	}
	else {
		cout << "Cannot create CSV file for saving. Printing it out now..." << endl;
		cout << endl;
		cout << "#Ferns,#Decisions,Mean_computation_time,SD_computation_time(ms),Mean_class_accuracy,SD_class_accuracy,Mean_MAAE,SD_MAAE" << endl;
		for (int pair = 0; pair < number_of_pairs; pair++) {
			int num_decisions = pairs_of_decisions_ferns[pair][0];
			int num_ferns = pairs_of_decisions_ferns[pair][1];

			cout << num_ferns << "," << num_decisions << "," << mean_computation_time[pair]*1000 << "," << sd_computation_time[pair]*1000 << "," <<
					mean_class_accuracy[pair] << "," << sd_class_accuracy[pair] << "," << mean_MAAE[pair] << "," << sd_MAAE[pair] << endl;
		}
	}

	int index_max_time, index_max_acc, index_max_error;
	int index_min_time, index_min_acc, index_min_error;

	double max_time = max(mean_computation_time, number_of_pairs, index_max_time);
	double min_time = min(mean_computation_time, number_of_pairs, index_min_time);
	double max_acc = max(mean_class_accuracy, number_of_pairs, index_max_acc);
	double min_acc = min(mean_class_accuracy, number_of_pairs, index_min_acc);
	double max_error = max(mean_MAAE, number_of_pairs, index_max_error);
	double min_error = min(mean_MAAE, number_of_pairs, index_min_error);
	//cout << "INDICES: " << index_max_time << ", " << index_min_time << ", " << index_max_acc << ", " << index_min_acc << ", " << index_max_error << ", " << index_min_error << endl;

	cout << "Max computation time " << max_time << " s with " << pairs_of_decisions_ferns[index_max_time][1] << " ferns, " << pairs_of_decisions_ferns[index_max_time][0] << " decisions." << endl;
	cout << "Min computation time " << min_time << " s with " << pairs_of_decisions_ferns[index_min_time][1] << " ferns, " << pairs_of_decisions_ferns[index_min_time][0] << " decisions." << endl;
	cout << "Max class accuracy " << max_acc << " % with " << pairs_of_decisions_ferns[index_max_acc][1] << " ferns, " << pairs_of_decisions_ferns[index_max_acc][0] << " decisions." << endl;
	cout << "Min class accuracy " << min_acc << " % with " << pairs_of_decisions_ferns[index_min_acc][1] << " ferns, " << pairs_of_decisions_ferns[index_min_acc][0] << " decisions." << endl;
	cout << "Max MAAE " << max_error << " degrees with " << pairs_of_decisions_ferns[index_max_error][1] << " ferns, " << pairs_of_decisions_ferns[index_max_error][0] << " decisions." << endl;
	cout << "Min MAAE " << min_error << " degrees with " << pairs_of_decisions_ferns[index_min_error][1] << " ferns, " << pairs_of_decisions_ferns[index_min_error][0] << " decisions." << endl;*/

	/*random_shuffle(file_list.begin(), file_list.end(), myrandom);
	for (unsigned int file_idx = 0; file_idx < file_list.size(); file_idx++) {
		ProcessFile(directory, file_list[file_idx]);
	}

	//fern_based_classifier * classifier;
	//classifier = new fern_based_classifier("train_data");
	fern_based_classifier * classifier;

	//ferns * Ferns;
	//Ferns = new ferns(5,10,0,0,0,0);

	vector<vector<float> > descriptors(training_data);
	//HOGDescriptor hog(Size(20,20), Size(10,10), Size(5,5), Size(5,5), 9);
	//cout << hog.getDescriptorSize() << endl;

	for (int i = 0; i < training_data; i++) {
		//string image_name = "__";		// TODO change to get files from directory
		//images[i] = imread(image_name);
		//resize(images[i], images[i], Size(20,20));

		hog.compute(images[i], descriptors[i]);
	}

	classifier = new fern_based_classifier(8,20,16);
	//for (int view = 0; view < 3; view++)
	//int view = 0;
	//classifier->Ferns->visualize(train_names[view], images[view], descriptors[view], ground_truths[view], ground_angles[view]);
	//return 0;

	//classifier->Ferns->drop(descriptors[0], images[0]);
	//return 0;

	classifier->train(descriptors, images, ground_truths, training_data);

	classifier->finalize_training();

	//int64 now = getTickCount();
	//double t = (double)(now-start)/getTickFrequency();
	//cout << t << endl;

	vector<float> test_descriptor;
	int64 sum_time = 0;
	int error = 0;
	int count_correct_class = 0;
	for (int i = 0; i < test_data; i++) {
		int64 start = cv::getTickCount();
		hog.compute(test_images[i], test_descriptor);
		int output_class;
		int output_angle = classifier->recognize_interpolate(test_descriptor, test_images[i], output_class);
		//int output_class = classifier->recognize(test_descriptor, test_images[i]);
		int64 now = cv::getTickCount();
		cout << test_angles[i] << " " << output_angle << "\t\t" << test_ground_truths[i] << " " << output_class << endl;
		int diff = abs(test_angles[i] - output_angle);
		if (diff > 180)
			error += 360 - diff;
		else
			error += diff;
		if (output_class == test_ground_truths[i])
			count_correct_class++;
		sum_time += now-start;
	}
	cout << double(sum_time)/getTickFrequency()/test_data << endl;
	cout << "Average test error: " << double(error)/double(test_data) << endl;
	cout << "Class correct: " << double(count_correct_class)/double(test_data) << endl;

	//classifier->print_distributions();

	cout << endl;

	// ******* Timed *******
	// start = cv::getTickCount();
	//hog.compute(img, descriptor);

	//classifier->train(descriptor, &img, ground_truths,1);
	//index = Ferns->drop(&descriptor[0], img);
	//int64 now = cv::getTickCount();
	// ******* Timed *******
	//double t = (double)(now-start)/cv::getTickFrequency();

	//std::cout << t << std::endl;*/

	//int64 total_finish = getTickCount();
	//cout << "Total searching Time " << double(total_finish-total_start)/getTickFrequency() << " s" << endl;

	//return 0;


	// ***** For finding the best combination of decisions *****
	int num_ferns = 200;		// TODO waiting for results
	int num_decisions = 16;		// TODO waiting for results

	int number_of_trials = 1000;			// Try number_of_trials times to find the best combination
	int number_of_random_dataset = 20;	// Randomly select training & testing set for number_of_random_dataset times.

	fern_based_classifier * best_classifier_acc, * best_classifier_MAAE;			// For the current best combination, using accuracy or MAAE as criteria
	best_classifier_acc = new fern_based_classifier(8,num_ferns,num_decisions,hog_size);
	best_classifier_MAAE = new fern_based_classifier(8,num_ferns,num_decisions,hog_size);
	// TODO OR KEEP FERNS ONLY?
	double best_acc1 = -100, best_MAAE1 = 360, sd_acc_best1, sd_MAAE_best1;		// for best_classifier_acc
	double best_acc2 = -100, best_MAAE2 = 360, sd_acc_best2, sd_MAAE_best2;		// for best_classifier_MAAE

	int64 start = getTickCount();
	for (int trial = 0; trial < number_of_trials; trial++) {
		double class_accuracy[number_of_random_dataset];
		double MAAE[number_of_random_dataset];
		fern_based_classifier * classifier;
		classifier = new fern_based_classifier(8,num_ferns,num_decisions,hog_size);
		if (trial % 10 == 0)
			cout << "Trial " << trial << endl;
		for (int it_dataset = 0; it_dataset < number_of_random_dataset; it_dataset++) {
			//if (it_dataset % 10 == 9)
			//	cout << "> Shuffle " << it_dataset+1 << endl;
			// Reshuffling images
			images.clear();
			ground_truths.clear();
			ground_angles.clear();
			train_names.clear();
			test_images.clear();
			test_ground_truths.clear();
			test_angles.clear();
			test_names.clear();
			training_data = 0;
			count_test = 0;
			classifier->reset_leaves_distributions();
			random_shuffle(file_list.begin(), file_list.end(), myrandom);
			for (unsigned int file_idx = 0; file_idx < file_list.size(); file_idx++) {
				ProcessFile(directory, file_list[file_idx], hog_size);
			}

			vector<vector<float> > descriptors(training_data);

			for (int i = 0; i < training_data; i++) {
				hog.compute(images[i], descriptors[i]);
			}

			// Training
			classifier->train(descriptors, images, ground_truths, training_data);
			classifier->finalize_training();

			// Testing
			vector<float> test_descriptor;
			int error = 0;
			int count_correct_class = 0;
			for (int i = 0; i < test_data; i++) {
				hog.compute(test_images[i], test_descriptor);
				int output_class, output_angle;
				classifier->recognize_interpolate(test_descriptor, test_images[i], output_class, output_angle);
				//cout << test_angles[i] << " " << output_angle << "\t\t" << test_ground_truths[i] << " " << output_class << endl;
				int diff = abs(test_angles[i] - output_angle);
				if (diff > 180)
					error += 360 - diff;
				else
					error += diff;
				//cout << error << ", ";
				//if (i%10 == 0)
				//	cout << endl;
				if (output_class == test_ground_truths[i])
					count_correct_class++;
			}

			class_accuracy[it_dataset] = double(count_correct_class)/double(test_data)*100.0;
			MAAE[it_dataset] = double(error)/double(test_data);
		}
		double mean_acc, mean_MAAE, sd_acc, sd_MAAE;

		mean_and_sd(class_accuracy, number_of_random_dataset, mean_acc, sd_acc);
		mean_and_sd(MAAE, number_of_random_dataset, mean_MAAE, sd_MAAE);

		//cout << "Accuracy " << mean_acc << "+-" << sd_acc << " %."<< endl;
		//cout << "MAAE " << mean_MAAE << "+-" << sd_MAAE << " deg." << endl;

		if (mean_acc > best_acc1) {
			// Comparing by accuracy
			// TODO copy the classifier to best_classifier (or copy ferns?)
			//delete best_classifier_acc;
			best_classifier_acc = new fern_based_classifier(classifier);
			best_acc1 = mean_acc;
			sd_acc_best1 = sd_acc;
			best_MAAE1 = mean_MAAE;
			sd_MAAE_best1 = sd_MAAE;
			cout << "> Found a better combination for accuracy in trial " << trial << "." << endl;
			cout << "> > New accuracy = " << best_acc1 << ". New MAAE = " << best_MAAE1 << endl;
		}
		//else {
		//	cout << "No accuracy improvement." << endl;
		//}

		if (mean_MAAE < best_MAAE2) {
			// Comparing by MAAE
			// TODO copy the classifier to best_classifier (or copy ferns?)
			//delete best_classifier_MAAE;
			best_classifier_MAAE = new fern_based_classifier(classifier);
			best_acc2 = mean_acc;
			sd_acc_best2 = sd_acc;
			best_MAAE2 = mean_MAAE;
			sd_MAAE_best2 = sd_MAAE;
			cout << "> Found a better combination for MAAE in trial " << trial << "." << endl;
			cout << "> > New MAAE = " << best_MAAE2 << ". New accuracy = " << best_acc2 << "." << endl;
		}
		//else {
		//	cout << "No MAAE improvement." << endl;
		//}

		delete classifier;

		// ??? Need to delete classifier ???
	}
	int64 finish = getTickCount();

	// Save the best one
	cout << "Finished " << number_of_trials << " trials in " << double(finish-start)/getTickFrequency() << " s." << endl;
	cout << "Saving the best classifier for accuracy ..." << endl;
	char file_acc[100];
    sprintf(file_acc,"classifier_acc_%d-%d",num_ferns,num_decisions);
	best_classifier_acc->save(file_acc);
	cout << "Saved." << endl;
	cout << "Saving the best classifier for MAAE ..." << endl;
	char file_MAAE[100];
	sprintf(file_MAAE,"classifier_MAAE_%d-%d",num_ferns,num_decisions);
	best_classifier_MAAE->save(file_MAAE);
	cout << "Saved." << endl;

	delete best_classifier_acc;
	delete best_classifier_MAAE;

	return 0;
	// *********************************************************

}

void ProcessDirectory(string directory, vector<string>& file_list) {
	string dir_to_open = path + directory;
	cout << dir_to_open << endl;
	DIR* dir = opendir(dir_to_open.c_str());

	path = dir_to_open + "/";

	if (NULL == dir) {
		cout << "Could not open directory: " << dir_to_open.c_str() << endl;
		return;
	}

	dirent* entity = readdir(dir);

	while (entity != NULL) {
		ProcessEntity(entity, file_list);
		entity = readdir(dir);
	}

	path.resize(path.length() - 1 - directory.length());
	closedir(dir);
}

void ProcessEntity(struct dirent* entity, vector<string>& file_list) {
	if (entity->d_type == DT_DIR) {
		if (entity->d_name[0] == '.')
			return;

		ProcessDirectory(string(entity->d_name), file_list);
		return;
	}

	if (entity->d_type == DT_REG) {
		if (entity->d_name[0] != '0')		// Not image file (README.txt)
			return;
		//ProcessFile(string(entity->d_name));
		//cout << string(entity->d_name) << ", " << endl;
		file_list.push_back(string(entity->d_name));
		return;
	}

	cout << "Not a file or a directory" << endl;
}

void ProcessFile(string dir, string file, int img_size) {
	string file_to_open = path + dir + '/' + file;
	char angle_c[3] = {file[9], file[10], file[11]};
	int angle = atoi(angle_c);
	Mat img = imread(file_to_open);
	resize(img, img, Size(img_size,img_size));
	int category;
	if (angle <= 22)
		category = 0;
	else if (angle <= 67)
		category = 1;
	else if (angle <= 112)
		category = 2;
	else if (angle <= 157)
		category = 3;
	else if (angle <= 202)
		category = 4;
	else if (angle <= 247)
		category = 5;
	else if (angle <= 292)
		category = 6;
	else if (angle <= 337)
		category = 7;
	else
		category = 0;
	/*if (angle <= 45)
		category = 0;
	else if (angle <= 90)
		category = 1;
	else if (angle <= 135)
		category = 2;
	else if (angle <= 180)
		category = 3;
	else if (angle <= 225)
		category = 4;
	else if (angle <= 270)
		category = 5;
	else if (angle <= 315)
		category = 6;
	else if (angle <= 360)
		category = 7;
	else
		category = 0;*/
	if (count_test < test_data) {
		test_ground_truths.push_back(category);
		test_images.push_back(img);
		test_angles.push_back(angle);
		test_names.push_back(file);
		count_test++;
		/*ground_truths.push_back(category);
		images.push_back(img);
		training_data++;*/
	}
	else {
		ground_truths.push_back(category);
		ground_angles.push_back(angle);
		images.push_back(img);
		train_names.push_back(file);
		training_data++;
	}
}

void mean_and_sd(double data[], size_t length, double& mean, double& sd) {
	double sum = 0;
	double sum_sq = 0;
	for (size_t i = 0; i < length; i++) {
		sum += data[i];
		sum_sq += data[i]*data[i];
	}
	mean = sum/double(length);
	sd = sqrt(sum_sq/length - mean*mean);
}

double max(double data[], size_t length, int& max_idx) {
	double max_val = data[0];
	max_idx = 0;
	for (size_t i = 0; i < length; i++) {
		if (data[i] > max_val) {
			max_idx = i;
			max_val = data[i];
		}
	}
	return max_val;
}

double min(double data[], size_t length, int& min_idx) {
	double min_val = data[0];
	min_idx = 0;
	for (size_t i = 0; i < length; i++) {
		if (data[i] < min_val) {
			min_idx = i;
			min_val = data[i];
		}
	}
	return min_val;
}
