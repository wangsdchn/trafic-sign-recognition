#include "Header.hpp"

#define winSize 16
#define winStep 8
#define scalefactor 1.25


void Recognition_(Mat src)
{
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
	feature_node *x_space = new feature_node[HOG_COLS + 1];
	HOGDescriptor hog(Size(HEIGHT, WIDTH), Size(HEIGHT / 4, WIDTH / 4), Size(HEIGHT / 8, WIDTH / 8), Size(HEIGHT / 8, WIDTH / 8), N_BINS);
	vector<float> descriptors;
	Rect rect;
	Mat roi;
	Mat roiLuvImg;
	Mat LuvImg;
	Mat dst(src.size(), src.type());
	dst = src;
	imwrite("./src.jpg", dst);
	double time0 = getTickCount();

	float scale = 1.0;
	float scoreth = 0;

	double* dec_values = new double[44];
	double score;
	int predictValue;

	Mat grayImg(src.size(), src.type());

	if (src.channels() == 3)
		cvtColor(src, grayImg, COLOR_BGR2GRAY);
	else
		grayImg = src;
	cvtColor(src, LuvImg, COLOR_BGR2Luv);
	cout << (getTickCount() - time0) / getTickFrequency() << endl;
	while (src.rows > 8*winSize && src.cols > 8*winSize)
	{
		cvtColor(src, grayImg, COLOR_BGR2GRAY);
		cvtColor(src, LuvImg, COLOR_BGR2Luv);
//		hog.compute(src , descriptors);
//		cout << (getTickCount() - time0) / getTickFrequency()<<" ";
		for (int y = 0; y < src.rows - winSize; y += winStep)
		{
			for (int x = 0; x < src.cols - winSize; x += winStep)
			{
				
				rect = Rect(x,y,winSize,winSize);
				roi = grayImg(rect);
				resize(roi,roi,Size(HEIGHT,WIDTH));
				hog.compute(roi, descriptors);
				// Predict with the SVM
				for (int j = 0; j < descriptors.size(); j++)
				{
					x_space[j].index = j + 1;
					x_space[j].value = descriptors[j];
				}
				
				
				roiLuvImg = LuvImg(rect);
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
						x_space[j + k*nc + descriptors.size()].index = j + k*nc + descriptors.size() + 1;
						x_space[j + k*nc + descriptors.size()].value = p[j];
					}
				}

				x_space[HOG_COLS - 1].index = HOG_COLS;
				x_space[HOG_COLS - 1].value = 0.5;
				x_space[HOG_COLS].index = -1;
				
				predictValue = predict_values(model_cir_tri, x_space, dec_values, score);
				if (predictValue == 0 && score>scoreth)
				{
					predictValue = predict_values(model_cir_red_blue, x_space, dec_values, score);
					if (predictValue == 1 && score>scoreth)
						predictValue = predict_values(model_cir_blue, x_space, dec_values, score);
					else if (predictValue == 0 && score>scoreth)
						predictValue = predict_values(model_cir_red, x_space, dec_values, score);
					else
						predictValue = -1;
				}
				else if (predictValue == 1 && score>scoreth)
				{
					predictValue = predict_values(model_tri, x_space, dec_values, score);
				}
				else
					predictValue = -1;
				
				if (predictValue >=0 && score>0.5)
				{
//					cout << predictValue << " " << score << endl;
					rectangle(dst, Rect(rect.x/scale,rect.y/scale,rect.height/scale,rect.width/
						scale), Scalar(0, 255, 0), 2, 8, 0);
					char str[20];
					sprintf(str, "%d %.3f", predictValue, score);
					putText(dst, str, Point(rect.x/scale, rect.y/scale), 1, 1.0, Scalar(255, 0, 0), 2);
				}
				
			}
		}
//		cout << (getTickCount() - time0) / getTickFrequency() << endl;
		scale = 1 / scalefactor;
		cout << src.rows << " " << src.cols << endl;
		resize(src,src,Size(src.cols*scale,src.rows*scale));
	}
	cout << (getTickCount() - time0) / getTickFrequency() << endl;
	imwrite("./dst.jpg",dst);
}
