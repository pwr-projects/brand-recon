#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>


/** @function main */
int main(int argc, char **argv)
{
    auto img1 = "../dataset/test_logo.png";
    auto img2 = "../dataset/test_shop.png";
    auto img_1 = cv::imread(img1, cv::IMREAD_GRAYSCALE);
    auto img_2 = cv::imread(img2, cv::IMREAD_GRAYSCALE);
    cv::Mat img_keypoints_1;
    cv::Mat img_keypoints_2;

    if (!img_1.data && !img_2.data)
    {
        std::cout << " --(!) Error reading images " << std::endl;
        return -1;
    }

    //-- Step 1: Detect the keypoints using SURF Detector
    auto minHessian = 400;
    // auto detector = cv::xfeatures2d::SIFT::create(minHessian);
    auto detector = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    drawKeypoints(img_1, keypoints_1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    drawKeypoints(img_2, keypoints_2, img_keypoints_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::imshow("Keypoints 1", img_keypoints_1);
    cv::imshow("Keypoints 2", img_keypoints_2);

    cv::waitKey(0);

    return 0;
}
