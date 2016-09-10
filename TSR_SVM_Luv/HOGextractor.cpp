
#include "Header.hpp"

void HOGextractor(const vector<string> &imgpath, vector<int> &imglable, feature_node *&x_space, problem &prob)
{
	HOGDescriptor hog(Size(HEIGHT, WIDTH), Size(HEIGHT / 4, WIDTH / 4), Size(HEIGHT / 8, WIDTH / 8), Size(HEIGHT / 8, WIDTH / 8), N_BINS);
	vector<float> descriptors;
	Mat srcImg;
	Mat grayImg;
	Mat dstImg;
	Mat LuvImg;
	
	for (int i = 0; i < imgpath.size(); i++)
	{
//	cerr<<i<<endl;
		string path=PATH_GTSRB+"GTSRB/train/"+imgpath[i];
		srcImg = imread(path);
		
		if (!srcImg.data){
			cout << "Load image failed" << endl;
			break;
		}
		
		resize(srcImg,dstImg,Size(HEIGHT,WIDTH));
		cvtColor(dstImg, grayImg, CV_BGR2GRAY);
		cvtColor(dstImg,LuvImg,COLOR_BGR2Luv);
		normalize(LuvImg,LuvImg,0.0,1.0,NORM_MINMAX);
//		normalize(LuvImg(1),LuvImg(1),1.0,0.0,NORM_MINMAX);
//		normalize(LuvImg(2),LuvImg(2),1.0,0.0,NORM_MINMAX);

		hog.compute(grayImg, descriptors);
		for (int j = 0; j < descriptors.size(); j++)
		{
			x_space[j + i* (HOG_COLS + 1)].index = j + 1;
			x_space[j + i* (HOG_COLS + 1)].value = descriptors[j];
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
				x_space[j + k*nc + descriptors.size() + i* (HOG_COLS + 1)].index = j + k*nc + descriptors.size() + 1;
				x_space[j + k*nc + descriptors.size() + i* (HOG_COLS + 1)].value = p[j];
			}
		}
				
		x_space[(HOG_COLS + 1)*(i + 1) - 2].index = HOG_COLS;
		x_space[(HOG_COLS + 1)*(i + 1) - 2].value = prob.bias;
		x_space[(HOG_COLS + 1)*(i + 1) - 1].index = -1;
		prob.x[i] = &x_space[(HOG_COLS + 1)*i];
		prob.y[i] = imglable[i];
		
	}	
}
