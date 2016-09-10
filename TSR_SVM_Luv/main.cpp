
#include "Header.hpp"
bool sort2(Rect left, Rect right){
	if (left.area() > right.area()) return 1;
	else return 0;
};
int main()
{	

//	SVM_train();
	/*
	SVM_test();
	
	string path = PATH_GTSRB+"GTSRB/pos.txt";
	ifstream Test(path.c_str());
	path =PATH_GTSRB+"GTSRB/test_pos_predict.txt";
	ifstream posPredictLable(path.c_str());
	path =PATH_GTSRB+"GTSRB/test_neg_predict.txt";
	ifstream negPredictLable(path.c_str());
	int n = 0;
	ComePare(Test, posPredictLable,negPredictLable,n);
	Test.close();
	posPredictLable.close();
	negPredictLable.close();
	cout << n << endl;
	*/
	string path1= PATH_GTSRB + "0001.png";
	Mat src;
	src = imread(path1);
	if (!src.data){
		cout << "Load Image Error" << endl;
		exit(1);
	}
	
	Recognition(src);
//	Recognition_(src);
	/*

		model *model_cir_tri = NULL;
		model *model_cir_red_blue = NULL;
		model *model_cir_red = NULL;
		model *model_cir_blue = NULL;
		model *model_tri = NULL;
		model_cir_tri = load_model(SVM_model_cir_tri);
		model_cir_red_blue = load_model(SVM_model_cir_red_blue);
		model_cir_red = load_model(SVM_model_cir_red);
		model_cir_blue = load_model(SVM_model_cir_blue);
		model_tri = load_model(SVM_model_tri);
		feature_node *x = new feature_node[HOG_COLS + 1];
		HOGDescriptor hog(Size(HEIGHT, WIDTH), Size(HEIGHT / 4, WIDTH / 4), Size(HEIGHT / 8, WIDTH / 8), Size(HEIGHT / 8, WIDTH / 8), N_BINS);
		vector<float> descriptors;
		Mat src;

		float fGamma = 1.5;
		float lut[256] = { 0 };
		float fGamma2 = 1 / 1.5;
		float lut2[256] = { 0 };
		for (int i = 0; i < 256; i++)
		{
			lut[i] = pow((float)(i / 255.0), fGamma) * 255.0f;
			lut2[i] = pow((float)(i / 255.0), fGamma2) * 255.0f;
		}

	VideoCapture cap("./tsr.avi");
	if (!cap.isOpened())
	{
		cout << "error" << endl;
		exit(1);
	}
	while (1)
	{

		double time0 = getTickCount();
		cap >> src;
		if (!src.data)
			continue;
		Mat grayImg(src.size(), src.type());

		if (src.channels() == 3)
			cvtColor(src, grayImg, COLOR_BGR2GRAY);
		else
			grayImg = src;

		Mat LuvImg;
		cvtColor(src, LuvImg, COLOR_BGR2Luv);
		//	imwrite("../Luv.png", LuvImg);
		cout << (getTickCount() - time0) / getTickFrequency() << endl;		

		Mat gray(src.size(), CV_8UC1);

		int nr = src.rows;
		int nc = src.cols;
		int channels = src.channels();
		if (src.isContinuous() && gray.isContinuous())
		{
			nr = 1;
			nc = nc*src.rows*channels;
		}
		uchar *p;
		uchar* p_gray;
		float tmp = 0.;
		float red = 0., blue = 0., yellow = 0.;
		float red_g = 0., blue_g = 0., green_g = 0.;
		for (int i = 0; i < nr; i++)
		{
			p = src.ptr<uchar>(i);
			p_gray = gray.ptr<uchar>(i);
			for (int j = 0; j < nc; j = j + channels)
			{
				blue_g = p[j];
				green_g = p[j + 1];
				red_g = p[j + 2];
				tmp = blue_g + green_g + red_g;	//B--0 , G--1, R--2
				//				red = red_g/tmp;
				//				blue = blue_g/tmp;
				red = MAX(0, MIN(red_g - green_g, red_g - blue_g) / tmp);
				blue = MAX(0, MIN(blue_g - red_g, blue_g - green_g) / tmp);
				yellow = MAX(0, MIN(red_g - blue_g, green_g - blue_g) / tmp);

				if (red>MAX(blue, yellow))
					p_gray[j / channels] = saturate_cast<int>(255 * red);	//saturate_cast<uchar>
				else if (blue>MAX(red, yellow))
					p_gray[j / channels] = saturate_cast<int>(255 * blue);
				else
					p_gray[j / channels] = saturate_cast<int>(255 * yellow);
			}
		}
		cout << (getTickCount() - time0) / getTickFrequency() << endl;

		vector<vector<Point> > regions;
		vector<Rect> boxs;
		MSER mserExtractor(5, 80, 3600, 0.35, 0.2, 200, 1.01, 0.003, 5);
		mserExtractor(gray, regions, Mat());

		cout << (getTickCount() - time0) / getTickFrequency() << endl;
		//	cerr << regions.size() << endl;

		vector<vector<Point> > contours_poly(regions.size());
		vector<Rect> boundRect(regions.size());

		vector<Rect> goodBoundRect;
		vector<vector<Point> > good_contours_poly;
		for (int i = 0; i < regions.size(); i++)
		{
			approxPolyDP(Mat(regions[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
//			rectangle(src, boundRect[i], Scalar(0, 255, 0));
			float a = (float)boundRect[i].height / boundRect[i].width;
			if (a<0.75 || a>1.25)
				continue;
			else
			{
				goodBoundRect.push_back(boundRect[i]);
			}
		}
		sort(goodBoundRect.begin(), goodBoundRect.end(), sort2);
		float thresh = 1000.0;
		float distance;
		cout << goodBoundRect.size() << endl;
		//		int tmp;
		set<int> erase_;
		for (int i = 0; i < goodBoundRect.size(); i++)
		{
			tmp = i;
			for (int j = i + 1; j < goodBoundRect.size(); j++){
				distance = pow(goodBoundRect[j].x - goodBoundRect[tmp].x, 2) + pow(goodBoundRect[j].y - goodBoundRect[tmp].y, 2);
				if (distance < goodBoundRect[j].height*goodBoundRect[j].height){
					erase_.insert(j);
				}
			}
		}
		cout << erase_.size() << endl;
		int n = 0;
		for (set<int>::iterator iter = erase_.begin(); iter != erase_.end(); iter++)
		{
			goodBoundRect.erase(goodBoundRect.begin() + (*iter - (n++)));

		}

		cout << goodBoundRect.size() << endl;
		Mat drawing = Mat::zeros(src.size(), CV_8UC3);
		Mat roi;
		int ret;
		double score;
		Mat roiLuvImg;
		double *dec_values = new double[44];
		for (int i = 0; i < goodBoundRect.size(); i++)
		{

			//		rectangle(src, goodBoundRect[i], Scalar(0, 255, 0), 2, 8, 0);
			roi = grayImg(goodBoundRect[i]);
			resize(roi, roi, Size(HEIGHT, WIDTH));
			hog.compute(roi, descriptors);
			// Predict with the SVM
			for (int j = 0; j < descriptors.size(); j++)
			{
				x[j].index = j + 1;
				x[j].value = descriptors[j];
			}

			roiLuvImg = LuvImg(goodBoundRect[i]);
			resize(roiLuvImg, roiLuvImg, Size(HEIGHT, WIDTH));
			normalize(roiLuvImg, roiLuvImg, 0.0, 1.0, NORM_MINMAX);
			int nr = roiLuvImg.rows;
			int nc = roiLuvImg.cols;
			int channels = roiLuvImg.channels();
			if (roiLuvImg.isContinuous())
			{
				nr = 1;
				nc = nc*roiLuvImg.rows*channels;
			}
			uchar *p;
			for (int k = 0; k < nr; k++)
			{
				p = roiLuvImg.ptr<uchar>(k);
				for (int j = 0; j < nc; j = j + 1)
				{
					x[j + k*nc + descriptors.size()].index = j + k*nc + descriptors.size() + 1;
					x[j + k*nc + descriptors.size()].value = p[j];
				}
			}

			x[HOG_COLS - 1].index = HOG_COLS;
			x[HOG_COLS - 1].value = 0.5;
			x[HOG_COLS].index = -1;
			ret = predict_values(model_cir_tri, x, dec_values, score);
			if (ret == 0)
			{
				ret = predict_values(model_cir_red_blue, x, dec_values, score);
				if (ret == 1)
					ret = predict_values(model_cir_blue, x, dec_values, score);
				else if (ret == 0)
					ret = predict_values(model_cir_red, x, dec_values, score);
				else
					ret = -1;
			}
			else if (ret == 1)
			{
				ret = predict_values(model_tri, x, dec_values, score);
			}
			else
				ret = -1;

			if (ret >= 0){
				cout << ret << " " << score << endl;
				rectangle(src, goodBoundRect[i], Scalar(0, 255, 0), 2, 8, 0);
				char str[512];
				sprintf(str, "%d %lf", ret, score);
				putText(src, str, Point(goodBoundRect[i].x, goodBoundRect[i].y), 1, 1.0, Scalar(255, 0, 0), 2);
			}
			//		drawContours(drawing, contours_poly, i, Scalar(0, 255, 0), 1, 8, vector<Vec4i>(), 0, Point());		
			//		circle(drawing, center[i], (int)radius[i], Scalar(0, 255, 0), 2, 8, 0);
		}
		imshow("video",src);
		waitKey(1);
		cout << (getTickCount() - time0) / getTickFrequency() << endl;
		//	imshow("mser", drawing);
		delete[] dec_values;
		dec_values = NULL;
		//	imshow("src",src);
		//		imwrite("./src.jpg", src);

		//		cout << (getTickCount() - time0) / getTickFrequency() << endl;
	}
	
	*/
system("pause");
	return 0;
}
