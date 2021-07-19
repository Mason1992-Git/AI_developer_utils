#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;

int main() {
    cv::Mat image = cv::imread("E:\\70-C++\\60-project\\01-MTCNN\\1.jpg",cv::IMREAD_GRAYSCALE);
    cv::imshow("picture",image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
