#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>


typedef struct _mouseInput
{
	cv::Point2i point;
	std::string windowName;
	int event;
}MouseInput;

void setMouseInput(MouseInput* mouseInput, int x, int y, const std::string& windowName, int event)
{
	mouseInput->point.x = x;
	mouseInput->point.y = y;
	mouseInput->windowName = windowName;
	mouseInput->event = event;
}
void onMouse1(int event, int x, int y, int, void* userData)
{
	MouseInput* mouseInput = (MouseInput*)userData;
	setMouseInput(mouseInput, x, y, "1st image", event);
}
void onMouse2(int event, int x, int y, int, void* userData)
{
	MouseInput* mouseInput = (MouseInput*)userData;
	setMouseInput(mouseInput, x, y, "2nd image", event);
}
cv::Mat getHistImage(int img, int patch, cv::Mat hHist, cv::Mat sHist, cv::Mat vHist, int histSize)
{
	int windowSize = 300;
	cv::Mat histImage(windowSize, windowSize, CV_8UC3, cv::Scalar(255, 255, 255));
	int binW = cvRound((double)histImage.cols / histSize);
	cv::Point before, current;

	cv::normalize(hHist, hHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(sHist, sHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(vHist, vHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < histSize; i++)
	{
		before = cv::Point(binW * (i - 1), histImage.rows - cvRound(hHist.at<float>(i - 1)));
		current = cv::Point(binW * i, histImage.rows - cvRound(hHist.at<float>(i)));
		cv::line(histImage, before, current, cv::Scalar(255, 0, 0), 2, 8, 0);

		before = cv::Point(binW * (i - 1), histImage.rows - cvRound(sHist.at<float>(i - 1)));
		current = cv::Point(binW * i, histImage.rows - cvRound(sHist.at<float>(i)));
		cv::line(histImage, before, current, cv::Scalar(0, 255, 0), 2, 8, 0);

		before = cv::Point(binW * (i - 1), histImage.rows - cvRound(vHist.at<float>(i - 1)));
		current = cv::Point(binW * i, histImage.rows - cvRound(vHist.at<float>(i)));
		cv::line(histImage, before, current, cv::Scalar(0, 0, 255), 2, 8, 0);
	}
	std::string text = "Image " + std::to_string(img) + " Patch " + std::to_string(patch) + "/4";
	cv::putText(histImage, text, cv::Point2i(0,20), 1, 1, cv::Scalar(0, 0, 0), 2, 8);


	std::cout << std::endl;
	return histImage;
}

typedef struct _compResult
{
	int patchIndex1;
	int patchIndex2;
	double value;
	bool operator<(const _compResult& other)
	{
		return value > other.value;
	}
}CompResult;

int main()
{
	double deltaTime = 1000 / 60 ;
	double scaleFactor = 0.2;
	const int patchCount = 4;

	std::string imgPaths[2] = { "assets/1st.jpg", "assets/2nd.jpg" };
	cv::Mat rgbImage[2];
	cv::Mat originalImage[2];
	cv::Mat resizedImage[2];
	std::string windowName[2] = { "1st image", "2nd image" };

	for (int i = 0; i < 2; i++)
	{
		rgbImage[i] = cv::imread(imgPaths[i]);
		cv::cvtColor(rgbImage[i], originalImage[i], cv::COLOR_BGR2HSV);
		cv::resize(rgbImage[i], resizedImage[i], cv::Size(), scaleFactor, scaleFactor);
		cv::imshow(windowName[i], resizedImage[i]);
	}

	MouseInput mouseInput;
	cv::setMouseCallback(windowName[0], onMouse1, &mouseInput);
	cv::setMouseCallback(windowName[1], onMouse2, &mouseInput);


	int method = 3;
	int histSize[] = { 16 };
	int hChannel[] = { 0 };
	int sChannel[] = { 1 };
	int vChannel[] = { 2 };
	float hRange_[] = { 0,180 };
	float svRange_[] = { 0,256 };
	const float* hRange[] = { hRange_ };
	const float* svRange[] = { svRange_ };
	cv::Size patchSize(140, 140);
	cv::Size scaledPatchSize(patchSize.width * scaleFactor, patchSize.height * scaleFactor);

	std::vector<cv::Rect> patchRects[2];

	for (int i = 0; i < 2; i++)
	{
		while (patchRects[i].size() < patchCount)
		{
			cv::waitKey(deltaTime);
			if (mouseInput.event != cv::EVENT_LBUTTONDOWN || mouseInput.windowName != windowName[i])
				continue;
			mouseInput.event = -1;

			cv::Rect rect(mouseInput.point / scaleFactor - cv::Point2i(patchSize / 2), patchSize);
			patchRects[i].push_back(rect);

			cv::Point2i rectCenter = mouseInput.point - cv::Point2i(scaledPatchSize / 2);
			std::cout << "(" << rectCenter.x << ", " << rectCenter.y << ")"<<std::endl;
			rect = cv::Rect(rectCenter, scaledPatchSize);
			cv::rectangle(resizedImage[i], rect, cv::Scalar(0, 0, 0),2);
			cv::putText(resizedImage[i], std::to_string(patchRects[i].size() - 1), rectCenter, 1, 2, cv::Scalar(0, 255, 0), 3, 8);
			
			cv::imshow(windowName[i], resizedImage[i]);
		}
	}

	cv::Mat patches[2][4];
	cv::Mat hists[2][4][3];

	std::vector<cv::Mat> histImages[2];

	for (int imgId = 0; imgId < 2; imgId++)
	{
		cv::Mat borderedImage;
		int borderSize = patchSize.height / 2;
		cv::copyMakeBorder(originalImage[imgId], borderedImage, borderSize, borderSize, borderSize, borderSize, cv::BORDER_REPLICATE);
		for (int patchId = 0; patchId < patchCount; patchId++)
		{
			cv::Rect rect = patchRects[imgId][patchId];
			cv::Rect movedRect(rect.x + borderSize, rect.y + borderSize, rect.width, rect.height);
			patches[imgId][patchId] = borderedImage(movedRect);

			cv::calcHist(&patches[imgId][patchId], 1, hChannel, cv::Mat(), hists[imgId][patchId][0], 1, histSize, hRange);
			cv::calcHist(&patches[imgId][patchId], 1, sChannel, cv::Mat(), hists[imgId][patchId][1], 1, histSize, svRange);
			cv::calcHist(&patches[imgId][patchId], 1, vChannel, cv::Mat(), hists[imgId][patchId][2], 1, histSize, svRange);
			histImages[imgId].push_back(getHistImage(imgId, patchId, hists[imgId][patchId][0], hists[imgId][patchId][1], hists[imgId][patchId][2], histSize[0]));
			for (int hsv = 0; hsv < 3; hsv++)
				cv::normalize(hists[imgId][patchId][hsv], hists[imgId][patchId][hsv], 0, 1, cv::NORM_MINMAX, CV_32F);

		}
	}
	cv::Mat img1, img2, allHistImage;
	cv::hconcat(histImages[0], img1);
	cv::hconcat(histImages[1], img2);
	cv::vconcat(img1, img2, allHistImage);
	cv::imshow("Histograms", allHistImage);
	
	std::vector<CompResult> compResults;
	for (int i = 0; i < patchCount; i++)
	{
		for (int j = 0; j < patchCount; j++)
		{
			double result = 0;
			for (int hsv = 0; hsv < 3; hsv++)
				result += cv::compareHist(hists[0][i][hsv], hists[1][j][hsv], method);
			if (method == 0 || method == 2)
				compResults.push_back({ i, j, result });
			else
				compResults.push_back({ i, j, -result });
			std::cout << i << "->" << j << " : " << compResults.back().value << std::endl;
		}
	}
	int matches[4] = { -1, -1, -1, -1 };
	double values[4] = { -1, -1, -1, -1 };
	std::vector<int> matched;
	std::sort(compResults.begin(), compResults.end());
	for (auto iter = compResults.begin(); iter != compResults.end(); iter++)
	{
		if (matches[iter->patchIndex1] == -1 &&
			std::find(matched.begin(), matched.end(), iter->patchIndex2) == matched.end())
		{
			matches[iter->patchIndex1] = iter->patchIndex2;
			values[iter->patchIndex1] = iter->value;
			matched.push_back(iter->patchIndex2);
		}
	}
	for (int i = 0; i < 4; i++)
	{
		std::cout << i << " -> " << matches[i] << " (" << values[i] << ")"<< std::endl;
	}
	cv::Mat resultImage;
	cv::hconcat(resizedImage[0], resizedImage[1], resultImage);

	for (int i = 0; i < patchCount; i++)
	{
		const cv::Rect& rect1 = patchRects[0][i];
		const cv::Rect& rect2 = patchRects[1][matches[i]];
		cv::Rect scaledRect1(rect1.x * scaleFactor, rect1.y * scaleFactor, rect1.width * scaleFactor, rect1.height * scaleFactor);
		cv::Rect scaledRect2(rect2.x * scaleFactor + resizedImage[0].cols, rect2.y * scaleFactor, rect2.width * scaleFactor, rect2.height * scaleFactor);
		cv::Point2i rectCenter1 = scaledRect1.tl() + cv::Point2i(scaledPatchSize) / 2;
		cv::Point2i rectCenter2 = scaledRect2.tl() + cv::Point2i(scaledPatchSize) / 2;

		cv::line(resultImage, rectCenter1, rectCenter2, cv::Scalar::all(0), 2, 8, 0);
	}
	cv::imshow("Result image", resultImage);
	cv::waitKey();
	return 0;
} 