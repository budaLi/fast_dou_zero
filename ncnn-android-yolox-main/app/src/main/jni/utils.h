//
// Created by lenovo on 2021/11/9.
//

#ifndef NCNN_ANDROID_NANODET_MASTER_UTILS_H
#define NCNN_ANDROID_NANODET_MASTER_UTILS_H
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct Point
{
    int x;
    int y;
};

class Utils {
public:
    int draw_unsupported(cv::Mat &rgb);
    int draw(cv::Mat& rgb, const std::vector<Object>& objects,const   std::vector<float> &class_scores,const std::vector<float>& landmarks,const std::vector<float>& poses,cv::Mat standard_face_cv,const  std::vector<float> face_attribute);
    int face_correction(cv::Mat &rgb, cv::Mat &standard_face_cv,std::vector<float> landmarks);
    int get_max_index(std::vector<float> output,int start,int end);
};


#endif //NCNN_ANDROID_NANODET_MASTER_UTILS_H
