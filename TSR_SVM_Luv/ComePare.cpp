
#include "Header.hpp"

void ComePare(ifstream &Test, ifstream &posPredictLable, ifstream &negPredictLable,int &n_diff)
{
	string buffer;
	vector<int> imglable;
	int lable;
	double score;

	while (!Test.eof())
	{
		Test >> buffer>>lable;
		imglable.push_back(lable);
	}
	
	vector<int> imglable1;
	while (!posPredictLable.eof())
	{
		posPredictLable >> buffer>> lable>>score;
		imglable1.push_back(lable);
	}
	
	vector<int> imglable2;
	while (!negPredictLable.eof())
	{
		negPredictLable >> buffer  >> lable>>score;
		imglable2.push_back(lable);
	}

	for (int i = 0; i < imglable1.size()-1; i++)
	{
		if (imglable1[i] != imglable[i]){
			if(imglable[i]>31){
			n_diff++;
			cout<<imglable[i]<<" "<<imglable1[i]<<endl;
			}
		}
	}
	int n_neg=0;
	for (int i = 0; i < imglable2.size()-1; i++)
		{
					if (imglable2[i] != -1)
						n_neg++;
	}
cout<<"n_diff: "<<n_diff<<"  n_neg: "<<n_neg<<endl;
}
