#include <time.h>
#include <iostream>
#include <opencv2\opencv.hpp>


using namespace std;
using namespace cv;

extern double	imgScale;
extern Size	patSize;

/*
 date:2016-04-09-16:05
 name:zhitong
 func:find the corners in the chessboard
*/
void cornerDetect(Mat m, vector<Point2f> &pts) {
	Mat		mGray, mask, simg;
	Size	camSize;

	if (m.empty()) {
		cout << "Matrix is empty!" << endl;
		return;
	}

	if (m.type() == CV_8UC3) {
		cvtColor(m, mGray, CV_BGR2GRAY);
	}
	else {
		mGray = m.clone();
	}

	camSize = mGray.size();

	clock_t flag1 = clock();
	resize(mGray, mask, Size(200, 200));
	GaussianBlur(mask, mask, Size(13, 13), 11);
	
	clock_t flag2 = clock();
	resize(mask, mask, camSize);
	medianBlur(mask, mask, 9);
	
	clock_t flag3 = clock();
	for (int v = 0; v < camSize.height; v++) {
		for (int u = 0; u < camSize.width; u++) {
			int x = (((int)mGray.at<uchar>(v, u) - (int)mask.at<uchar>(v, u)) << 1) + 128;
			mGray.at<uchar>(v, u) = max(min(x, 255), 0);
		}
	}
	resize(mGray, simg, Size(), imgScale, imgScale);

	clock_t flag4 = clock();
	bool found = findChessboardCorners(simg, patSize, pts);
	
	clock_t flag5 = clock();
	for (unsigned int i = 0; i < pts.size(); ++i) {
		pts[i] *= 1. / imgScale;
	}

	clock_t flag6 = clock();
	if (found) {
		cornerSubPix(mGray, pts, Size(21, 21), Size(-1, -1),
			TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 5000, 0.0001));
	}

	clock_t flag7 = clock();
	drawChessboardCorners(m, patSize, Mat(pts), found);

	clock_t flag8 = clock();
	cout << saturate_cast<double>(flag2 - flag1) / CLOCKS_PER_SEC << '\t'
		<< saturate_cast<double>(flag3 - flag2) / CLOCKS_PER_SEC << '\t'
		<< saturate_cast<double>(flag4 - flag3) / CLOCKS_PER_SEC << '\t'
		<< saturate_cast<double>(flag5 - flag4) / CLOCKS_PER_SEC << '\t'
		<< saturate_cast<double>(flag7 - flag6) / CLOCKS_PER_SEC << '\t'
		<< saturate_cast<double>(flag8 - flag7) / CLOCKS_PER_SEC << '\t'
		<< saturate_cast<double>(flag8 - flag1) / CLOCKS_PER_SEC << '\t' << found << '\t';

	/*ofstream file("imgPts.txt", ostream::out);
	file << Mat(pts);
	file.close();*/

	imshow("cornerDetect", m);
	waitKey(100);
	return;
}
