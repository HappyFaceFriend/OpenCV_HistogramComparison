#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <ctime>

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
	double deltaTime = 1000 / 60;
	double scaleFactor = 0.2;
	const int patchCount = 4;
	int randomRange = 10;

	std::string imgPaths[2] = { "assets/1st.jpg", "assets/2nd.jpg" };
	cv::Mat rgbImage[2];
	cv::Mat originalImage[2];
	cv::Mat resizedImage[2];
	std::string windowName[2] = { "1st image", "2nd image" };

	for (int i = 0; i < 2; i++)
	{
		rgbImage[i] = cv::imread(imgPaths[i]);
		cv::cvtColor(rgbImage[i], originalImage[i], cv::COLOR_BGR2HSV);
	}

	MouseInput mouseInput;

	std::mt19937 gen(42);
	std::uniform_int_distribution<int> rand(-randomRange, randomRange);

	cv::Point2i basePatches[2][4] = {
		{{264, 136}, {16, 370}, {336, 714}, {588, 472}},
		{{503, 213}, {145, 209}, {122, 702}, {503, 721}}
	};
	const int totalTestCase = 50;
	cv::Point2i randomPoints[totalTestCase];
	for (int i = 0; i < totalTestCase; i++)
		randomPoints[i] = cv::Point2i(rand(gen), rand(gen));
	float hRange[] = { 0,180 };
	float svRange[] = { 0,256 };

	int channels[] = { 0, 1 };
	const float* ranges[] = { hRange, svRange };

	int patchSize_ = 50;
	int method = 3;
	for (int ti = 16; ti <= 40; ti += 8)
	{
		std::vector<float> times;
		std::vector<float> accuracies;
		int binSize;
		for (patchSize_ = 50; patchSize_ <= 200; patchSize_ += 10)
		{
			binSize = ti;
			int histSize[] = { binSize, binSize };
			//std::cout << "------------------------------------------" << std::endl;
			clock_t startTime = clock();
			int correct = 0;
			cv::Size patchSize(patchSize_, patchSize_);
			cv::Size scaledPatchSize(patchSize.width * scaleFactor, patchSize.height * scaleFactor);
			for (int testcase = 0; testcase < totalTestCase; testcase++)
			{
				std::vector<cv::Rect> patchRects[2];
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						mouseInput.point = basePatches[i][j] + randomPoints[testcase];

						cv::Rect rect(mouseInput.point / scaleFactor - cv::Point2i(patchSize / 2), patchSize);
						patchRects[i].push_back(rect);

					}
				}
				cv::Mat patches[2][4];
				cv::Mat hists[2][4];

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

						cv::calcHist(&patches[imgId][patchId], 1, channels, cv::Mat(), hists[imgId][patchId], 2, histSize, ranges);
						cv::normalize(hists[imgId][patchId], hists[imgId][patchId], 0, 1, cv::NORM_MINMAX, CV_32F);
					}
				}

				std::vector<CompResult> compResults;
				for (int i = 0; i < patchCount; i++)
				{
					for (int j = 0; j < patchCount; j++)
					{
						if (method == 0 || method == 2)
							compResults.push_back({ i, j, cv::compareHist(hists[0][i], hists[1][j], method) });
						else
							compResults.push_back({ i, j, -cv::compareHist(hists[0][i], hists[1][j], method) });
					}
				}
				int matches[4] = { -1, -1, -1, -1 };
				double values[4] = { -1, -1, -1, -1 };
				std::vector<int> matched;
				std::sort(compResults.begin(), compResults.end());
				int cnt = 0;
				for (auto iter = compResults.begin(); iter != compResults.end(); iter++)
				{
					if (matches[iter->patchIndex1] == -1 &&
						std::find(matched.begin(), matched.end(), iter->patchIndex2) == matched.end())
					{
						matches[iter->patchIndex1] = iter->patchIndex2;
						matched.push_back(iter->patchIndex2);
						if (iter->patchIndex1 == iter->patchIndex2)
							cnt++;
					}
				}
				correct += cnt;
				//std::cout << "Test "<<testcase<<" : " << cnt <<"/4" << std::endl;

			}
			//std::cout << "Patch Size : " << patchSize_ << std::endl;
			//std::cout << "Channel : H, S, V" << std::endl;
			//std::cout << "Bin Size : " << binSize << std::endl;
			//std::cout << "Comparison Method : " << method << std::endl;
			//std::cout << "Average Accuracy : " << (float)correct / (totalTestCase * 4) * 100 << "%" << std::endl;
			//std::cout << "Time : " << (double)clock() - startTime << "ms" << std::endl;
			accuracies.push_back((float)correct / (totalTestCase * 4) * 100);
			times.push_back(clock() - startTime);
		}
		std::cout << "Bin Size : " << binSize << std::endl;
		std::cout << "Comparison Method : " << method << std::endl;
		std::cout << "[";
		for (int i = 0; i < accuracies.size(); i++)
		{
			std::cout << accuracies[i] << ",";
		}
		std::cout << "]" << std::endl;
		std::cout << "[";
		for (int i = 0; i < times.size(); i++)
		{
			std::cout << times[i] << ",";
		}
		std::cout << "]" << std::endl;

	}
	return 0;
}
