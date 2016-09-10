#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include "./liblinear/linear.h"
#include "./liblinear/tron.h"


using namespace std;
using namespace cv;

//#define SVM_model "./model/svm_modle_C0.2_32_43_cir_tri_neg_Luv1.txt"
//#define SVM_model "./model/svm_modle_C0.2_32_43_cir_red_blue_Luv1.txt"
//#define SVM_model "./model/svm_modle_C0.2_32_43_cir_red_Luv1.txt"
#define SVM_model "./model/svm_modle_C0.2_32_43_cir_blue_Luv1.txt"
//#define SVM_model "./model/svm_modle_C0.2_32_43_tri_Luv1.txt"
const string PATH_GTSRB = "E:/projests/TSR/test_images/";
//#define SVM_model "svm_modle_C10_64.txt"
#define HEIGHT 32
#define WIDTH 32

#define HOG_COLS 4837		//1764+32*32*3+1
#define N_BINS 9
void ImgPathRead(ifstream &datain, vector<string> &imgpath, vector<int> &imglable);
void HOGextractor(const vector<string> &imgpath, vector<int> &imglable, feature_node *&x_space, problem &prob);
void SVM_train();
void SVM_test();
void ComePare(ifstream &Test, ifstream &posPredictLable, ifstream &negPredictLable,int &n_diff);
void Recognition(Mat src);
void Recognition_(Mat src);


template < typename T >
void WriteToFile_Log(T &str)
{
	ofstream log_out("./model/output.txt", ios::out | ios::app);
	log_out << str << endl;
	log_out.close();
};

#define SVM_model_cir_tri		 "./model/svm_modle_C0.2_32_43_cir_tri_neg_Luv1.txt"
#define SVM_model_cir_red_blue	"./model/svm_modle_C0.2_32_43_cir_red_blue_Luv1.txt"
#define SVM_model_cir_red		 "./model/svm_modle_C0.2_32_43_cir_red_Luv1.txt"
#define SVM_model_cir_blue		 "./model/svm_modle_C0.2_32_43_cir_blue_Luv1.txt"
#define SVM_model_tri			 "./model/svm_modle_C0.2_32_43_tri_Luv1.txt"