
#include "Header.hpp"

void SVM_train()
{
	

	vector<string> imgpath;
	vector<int> imglable;
//	string path = PATH_GTSRB+"GTSRB/train_lable/train_cir_tri_neg.txt";
//	string path = PATH_GTSRB+"GTSRB/train_lable/train_cir_red_blue.txt";
//	string path = PATH_GTSRB+"GTSRB/train_lable/train_cir_red.txt";
	string path = PATH_GTSRB+"GTSRB/train_lable/train_cir_blue.txt";
//	string path = PATH_GTSRB+"GTSRB/train_lable/train_tri.txt";
	ifstream data_train(path.c_str());
	ImgPathRead(data_train, imgpath,imglable);
	data_train.close();
	
	int NSet=imgpath.size();
	cout << NSet << endl;
	int m = NSet*(HOG_COLS + 1);
	cout<<m<<endl;
//	feature_node *x_space = (feature_node*)malloc(sizeof(feature_node)*m);
	feature_node *x_space = new feature_node[m];

	problem prob = {0,0,0,0,0};
	prob.l = NSet;
	prob.n = HOG_COLS+1;
	prob.y = new double[NSet];
	prob.x=new feature_node *[NSet];
	prob.bias = 0.5;

	if (NULL == x_space){
		cout << "x_space error" << endl;
		exit(1);
	}
	if (NULL == prob.x){
		cout << "prob.x error" << endl;
		exit(1);
	}
	if (NULL == prob.y){
		cout << "prob.y error" << endl;
		exit(1);
	}

	cout << "Begin Train Hog  "<< endl;	
	HOGextractor(imgpath,imglable, x_space, prob);
	cout << "End Train Hog  " << endl;
	

/*	int weight_lable1[43]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42};
	double weight1[43]=
{10,1,1,2,1, 1,5,2,2,2, 1,2,1,1,3, 3,5,2,2,10, 7,7,7,4,8, 1,3,8,4,8, 5,3,8,3,5, 2,5,10,1,7, 6,8,8 };
*/
	int64 time0 = getTickCount();
	parameter param;
	param.C = 0.2;
	param.eps = FLT_EPSILON;
	param.p = 0.1;
	param.nr_weight =0; //43;
	param.weight_label = NULL;//weight_lable1;
	param.weight = NULL;//weight1;
	param.init_sol = NULL;
	param.solver_type = L2R_L2LOSS_SVC_DUAL;

//	double best_c;
//	double best_rate;
//	find_parameter_C(&prob,&param,5,0.1,1,&best_c,&best_rate);

	//WriteToFile_Log(best_c);
	//WriteToFile_Log(best_rate);
	//param.C = best_c;

	cout << "Begin Train " << endl;

	model *model=NULL;
	model = train(&prob, &param);
	save_model(SVM_model, model);
//-------------------------------
	cout << "End Train " << endl;
	WriteToFile_Log("SVM train time:");
	double time1 = (getTickCount() - time0) / getTickFrequency();
	WriteToFile_Log(time1);
	cout << time1 << endl;

	free_and_destroy_model(&model);
	destroy_param(&param);
	model = NULL;
	if(x_space==NULL){
		free(x_space);
		x_space = NULL;
	}
	if(prob.x==NULL){
		delete[] prob.x;
		prob.x = NULL;
	}
	if(prob.y==NULL){
		delete[] prob.y;
		prob.y = NULL;
	}
}
