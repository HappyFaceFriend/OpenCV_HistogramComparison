# OpenCV_HistogramComparison
Image comparison by image histograms using OpenCV (Computer vision class assignment)


## How to run
Run demo/Project1.exe

#How to modify
1. Import src/main.cpp to your opencv project and open it.
2. To change image, modify `imgPaths`
```
std::string imgPaths[2] = { "assets/1st.jpg", "assets/2nd.jpg" };
```
3. To change scale of image displayed, modify `scaleFactor`
```
double scaleFactor = 0.2;
```
4. To change method, channels, patch size, or bin size, change variables right before the for loop.
```
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
```

* autoeval.cpp is used for iterating tests to find the right parameters.

## Test result 1
![image](https://user-images.githubusercontent.com/11360981/160683749-ed3350be-e06e-48db-986b-b0e13b652425.png)
## Histograms of test result 1
![image](https://user-images.githubusercontent.com/11360981/160683815-a3bdb9bc-e501-453c-83ba-5cae3448e13b.png)

## Test result 2
![image](https://user-images.githubusercontent.com/11360981/160684092-857deb52-d76d-40d0-b0fa-a9f363653fef.png)


