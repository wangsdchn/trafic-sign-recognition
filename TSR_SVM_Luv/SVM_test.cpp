
#include "Header.hpp"

void SVM_test()
{
	vector<string> imgpath;
	vector<int> imglable1;
	string path;
	int64 time0 = getTickCount();
	
	model *model_cir_tri 		= NULL;
	model *model_cir_red_blue 	= NULL;
	model *model_cir_red 		= NULL;
	model *model_cir_blue 		= NULL;
	model *model_tri 				= NULL;
	model_cir_tri 			= load_model(SVM_model_cir_tri);
	model_cir_red_blue 		= load_model(SVM_model_cir_red_blue);
	model_cir_red 			= load_model(SVM_model_cir_red);
	model_cir_blue			= load_model(SVM_model_cir_blue);
	model_tri 				= load_model(SVM_model_tri);
	
	path = PATH_GTSRB+"GTSRB/neg.txt";
	ifstream data_test(path.c_str());
	ImgPathRead(data_test, imgpath, imglable1);
	data_test.close();

	cout << "Begin Test HOG" << endl;
	Mat srcImg;
	Mat LuvImg;
	Mat grayImg;
	Mat dstImg;
	HOGDescriptor hog(Size(HEIGHT, WIDTH), Size(HEIGHT / 4, WIDTH / 4), Size(HEIGHT / 8, WIDTH / 8), Size(HEIGHT / 8, WIDTH / 8), N_BINS);
	vector<float> descriptors;
	cout << "Begin  Neg Test" << endl;
	char line[512];
	
	path = PATH_GTSRB+"GTSRB/test_neg_predict.txt";
	ofstream predict_txt(path.c_str());
	feature_node *x = new feature_node[HOG_COLS+1];
	int ret=0;
	double score=0.0;
	
	cout << imgpath.size() << endl;
	for (int i = 0; i < imgpath.size(); i++)
	{
		string path=PATH_GTSRB+"GTSRB/val/"+imgpath[i];
		srcImg = imread(path);
		
		resize(srcImg, dstImg, Size(HEIGHT, WIDTH));
		cvtColor(dstImg, grayImg, CV_BGR2GRAY);
		cvtColor(dstImg,LuvImg,COLOR_BGR2Luv);
		normalize(LuvImg, LuvImg, 0.000001, 1.0, NORM_MINMAX);
		hog.compute(grayImg, descriptors);

		for (int j = 0; j < descriptors.size(); j++)
		{
			x[j].index = j+1;
			x[j].value = descriptors[j];
		}
		int nr = LuvImg.rows;
		int nc = LuvImg.cols;
		int channels = LuvImg.channels();
		if (LuvImg.isContinuous())
		{
			nr = 1;
			nc = nc*LuvImg.rows*channels;
		}
		uchar *p;
		for (int k = 0; k < nr; k++)
		{
			p = LuvImg.ptr<uchar>(k);
			for (int j = 0; j < nc; j = j + 1)
			{
				x[j + k*nc + descriptors.size()].index = j + k*nc + descriptors.size() + 1;
				x[j + k*nc + descriptors.size()].value = p[j];
			}
		}
		
		x[HOG_COLS-1].index = HOG_COLS;
		x[HOG_COLS-1].value = 0.5;
		x[HOG_COLS].index = -1;
		
		double *dec_values=new double[44];
		ret = predict_values(model_cir_tri,x,dec_values,score);
		if(ret==0)
		{
			ret = predict_values(model_cir_red_blue,x,dec_values,score);
			if(ret==0)
				ret = predict_values(model_cir_red,x,dec_values,score);
			else if(ret==1)
				ret = predict_values(model_cir_blue,x,dec_values,score);
			else
				ret=-1;
		}
		else if(ret==1)
		{
			ret = predict_values(model_tri,x,dec_values,score);
		}
		else
			ret = -1;

		sprintf(line, "%s %d %f\n", imgpath[i].c_str(), ret, score);
		predict_txt << line;
		
		delete []dec_values;
	}
	predict_txt.close();
	imgpath.clear();
	imglable1.clear();
	path = PATH_GTSRB+"GTSRB/pos.txt";
	data_test.open(path.c_str());
	ImgPathRead(data_test, imgpath, imglable1);
	data_test.close();

	cout << "Begin  Pos Test" << endl;

	
	path = PATH_GTSRB+"GTSRB/test_pos_predict.txt";
 predict_txt.open(path.c_str());

	
	cout << imgpath.size() << endl;
	for (int i = 0; i < imgpath.size(); i++)
	{
		string path=PATH_GTSRB+"GTSRB/val/"+imgpath[i];
		srcImg = imread(path);
		
		resize(srcImg, dstImg, Size(HEIGHT, WIDTH));
		cvtColor(dstImg, grayImg, CV_BGR2GRAY);
		cvtColor(dstImg,LuvImg,COLOR_BGR2Luv);
		normalize(LuvImg, LuvImg, 0.000001, 1.0, NORM_MINMAX);
		hog.compute(grayImg, descriptors);

		for (int j = 0; j < descriptors.size(); j++)
		{
			x[j].index = j+1;
			x[j].value = descriptors[j];
		}
		int nr = LuvImg.rows;
		int nc = LuvImg.cols;
		int channels = LuvImg.channels();
		if (LuvImg.isContinuous())
		{
			nr = 1;
			nc = nc*LuvImg.rows*channels;
		}
		uchar *p;
		for (int k = 0; k < nr; k++)
		{
			p = LuvImg.ptr<uchar>(k);
			for (int j = 0; j < nc; j = j + 1)
			{
				x[j + k*nc + descriptors.size()].index = j + k*nc + descriptors.size() + 1;
				x[j + k*nc + descriptors.size()].value = p[j];
			}
		}
		
		x[HOG_COLS-1].index = HOG_COLS;
		x[HOG_COLS-1].value = 0.5;
		x[HOG_COLS].index = -1;
		double *dec_values=new double[44];
		ret = predict_values(model_cir_tri,x,dec_values,score);
		if(ret==0)
		{
			ret = predict_values(model_cir_red_blue,x,dec_values,score);
			if(ret==0)
				ret = predict_values(model_cir_red,x,dec_values,score);
			else if(ret==1)
				ret = predict_values(model_cir_blue,x,dec_values,score);
			else
				ret=-1;
		}
		else if(ret==1)
		{
			ret = predict_values(model_tri,x,dec_values,score);
		}
		else
			ret = -1;

		sprintf(line, "%s %d %f\n", imgpath[i].c_str(), ret, score);
		predict_txt << line;
		
		delete []dec_values;
	}
	predict_txt.close();


//-----------------------------------------------
	cout << "End Test" << endl;
	WriteToFile_Log("SVM Test Time:");
	double time2 = (getTickCount() - time0) / getTickFrequency();
	WriteToFile_Log(time2);
	cout << time2 << endl;

}
