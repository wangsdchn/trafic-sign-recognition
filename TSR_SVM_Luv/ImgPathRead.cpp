
#include "Header.hpp"

void ImgPathRead(ifstream &datain, vector<string> &imgpath, vector<int> &imglable)
{
	string buffer;
	int lable;
	while (!datain.eof())
	{
		datain >> buffer>>lable;
		imgpath.push_back(buffer);
		imglable.push_back(lable);
	}
}
