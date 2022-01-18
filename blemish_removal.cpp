#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat img, lastStep, imgCopy;		
int r = 10;				// patch radius for square patch

// find boundaries of a valid patch after mouse click
vector<int> findPatchBoundaries(Mat img, int x, int y, int r) {
	int minX, maxX, minY, maxY;

	// topleft corner
	if (x - r < 0 && y - r < 0) {
		minX = 0;
		maxX = 2 * r;
		minY = 0;
		maxY = 2 * r;
	}
	// left edge
	else if (x - r < 0 && y - r >= 0 && y + r < img.size().height) {
		minX = 0;
		maxX = 2 * r;
		minY = y - r;
		maxY = y + r;
	}
	// lowerleft corner
	else if (x - r < 0 && y + r >= img.size().height) {
		minX = 0;
		maxX = 2 * r;
		maxY = img.size().height - 1;
		minY = maxY - 2 * r;
	}
	// bottom edge
	else if (x - r >= 0 && x + r < img.size().width && y + r >= img.size().height) {
		minX = x - r;
		maxX = x + r;
		maxY = img.size().height - 1;
		minY = maxY - 2 * r;
	}
	// lowerright corner
	else if (x + r >= img.size().width && y + r >= img.size().height) {
		maxX = img.size().width - 1;
		minX = maxX - 2 * r;
		maxY = img.size().height - 1;
		minY = maxY - 2 * r;
	}
	// right edge
	else if (x + r >= img.size().width && y - r >= 0 && y + r < img.size().height) {
		maxX = img.size().width - 1;
		minX = maxX - 2 * r;
		minY = y - r;
		maxY = y + r;
	}
	// upperright corner
	else if (x + r >= img.size().width && y - r < 0) {
		maxX = img.size().width - 1;
		minX = maxX - 2 * r;
		minY = 0;
		maxY = 2 * r;
	}
	// top edge
	else if (x - r >= 0 && x + r < img.size().width && y - r < 0) {
		minX = x - r;
		maxX = x + r;
		minY = 0;
		maxY = 2 * r;
	}
	// patch not close to any edge/corner
	else {
		minX = x - r;
		maxX = x + r;
		minY = y - r;
		maxY = y + r;
	}
	
	return vector<int>{minX, maxX, minY, maxY};
}

Mat cropPatch(Mat img, int x, int y, int r) {
	vector<int> patchboundaries = findPatchBoundaries(img, x, y, r);
	return img(Range(patchboundaries[2], patchboundaries[3]), Range(patchboundaries[0], patchboundaries[1]));
}

Point findSeamlessCloneCenter(Mat img, int x, int y) {
	vector<int> patchboundaries = findPatchBoundaries(img, x, y, r);

	int centerX = (patchboundaries[0] + patchboundaries[1]) / 2;
	int centerY = (patchboundaries[2] + patchboundaries[3]) / 2;

	return Point(centerX, centerY);
}

Mat findBestPatch(Mat img, int x, int y, int r) {
	vector<Mat> neighborPatches;

	// patchUp
	if (y - 2 * r >= 0) {
		Mat patchUp = cropPatch(img, x, y - 2 * r, r);
		neighborPatches.push_back(patchUp);					
	}
	// patchDown
	if (y + 2 * r < img.size().height) {
		Mat patchDown = cropPatch(img, x, y + 2 * r, r);
		neighborPatches.push_back(patchDown);
	}
	// patchLeft
	if (x - 2 * r >= 0) {
		Mat patchLeft = cropPatch(img, x - 2 * r, y, r);
		neighborPatches.push_back(patchLeft);
	}
	// patchRight
	if (x + 2 * r < img.size().width) {
		Mat patchRight = cropPatch(img, x + 2 * r, y, r);
		neighborPatches.push_back(patchRight);
	}
	// patchUpperLeft
	if (x - 2 * r >= 0 && y - 2 * r >= 0) {
		Mat patchUpperLeft = cropPatch(img, x - 2 * r, y - 2 * r, r);
		neighborPatches.push_back(patchUpperLeft);
	}
	// patchUpperRight
	if (x + 2 * r < img.size().width && y - 2 * r >= 0) {
		Mat patchUpperRight = cropPatch(img, x + 2 * r, y - 2 * r, r);
		neighborPatches.push_back(patchUpperRight);
	}
	// patchLowerLeft
	if (x - 2 * r >= 0 && y + 2 * r < img.size().height) {
		Mat patchLowerLeft = cropPatch(img, x - 2 * r, y + 2 * r, r);
		neighborPatches.push_back(patchLowerLeft);
	}
	// patchLowerRight
	if (x + 2 * r < img.size().width && y + 2 * r < img.size().height) {
		Mat patchLowerRight = cropPatch(img, x + 2 * r, y + 2 * r, r);
		neighborPatches.push_back(patchLowerRight);
	}

	double score, minScore = DBL_MAX;
	int minPatchInd = 0;
	for (int i = 0; i < neighborPatches.size(); i++) {
		Mat pGray;
		cvtColor(neighborPatches[i], pGray, COLOR_BGR2GRAY);

		// use L1-norm of the Sobel gradient to measure variations of neighboring patches. Select the patch with least variation
		Mat gx, gy;
		Sobel(pGray, gx, CV_32F, 1, 0);
		Sobel(pGray, gy, CV_32F, 0, 1);
		score = sum(abs(gx))[0] + sum(abs(gy))[0];
		if (score < minScore) {
			minScore = score;
			minPatchInd = i;
		}
	}
	return neighborPatches[minPatchInd];
}

// callback function for removing blemish
void onClickRemoveBlemish(int action, int x, int y, int flags, void* userdata) {
	if (action == EVENT_LBUTTONDOWN) {
		// save the result from previous step to a separate image
		lastStep = img.clone();

		// select the patch centered at the cursor	
		Mat patch = cropPatch(img, x, y, r);

		// find another patch with the least gradient variation from neighboring patches
		Mat bestPatch = findBestPatch(img, x, y, r);

		// replace the current patch with the new patch using seamless cloning
		Mat mask = Mat(bestPatch.size().height, bestPatch.size().width, CV_8U, Scalar(255));
		Point center = findSeamlessCloneCenter(img, x, y);
		seamlessClone(bestPatch, img, mask, center, img, NORMAL_CLONE);

		// display the image
		imshow("blemishRemovalResult", img);
	}
}

int main() {
	cout << "Using patch radius: " << r << endl;
	cout << "Press u to undo the last step." << endl;
	cout << "Press c to start over again." << endl;
	cout << "Press ESC to exit." << endl;

	img = imread("blemish.png");
	lastStep = img.clone();
	imgCopy = img.clone();
	
	namedWindow("blemishRemovalResult", WINDOW_NORMAL);
	setMouseCallback("blemishRemovalResult", onClickRemoveBlemish);

	int k = 0;
	while (k != 27) {
		imshow("blemishRemovalResult", img);
		k = waitKey();
		
		// press c to start over again
		if (k == 99) {
			imgCopy.copyTo(img);
		}

		// press u to undo the last step
		if (k == 117) {
			lastStep.copyTo(img);
		}
	}

	return 0;
}
