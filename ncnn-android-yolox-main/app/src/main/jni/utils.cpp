//
// Created by lenovo on 2021/11/9.
//
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>

#include "yolo_dou.h"

#include "cpu.h"
#include "utils.h"

int Utils::draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

int Utils::draw(cv::Mat &rgb, const std::vector<Object> &objects,const std::vector<float> &class_scores,const std::vector<float> &landmarks,const std::vector<float> &poses,cv::Mat standard_face_cv, std::vector<float> face_attribute) {

    //人脸框
    {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  face start ...");
        static const char* class_names[] = {
                "face"
        };

        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];

            fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

            cv::rectangle(rgb, obj.rect, cv::Scalar(255, 0, 0));

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > rgb.cols)
                x = rgb.cols - label_size.width;

            cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        }

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  face over ...");
    }

    const Object& obj = objects[0];
    // 活体检测
    {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  face live start ...");
        int topk = 3; //三个类别：假人脸（2D）、真人脸、假人脸（3D）
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int k = 0; k < topk; k++)
        {
            float score = class_scores[k];
            if (score > class_score)
            {
                class_index = k;
                class_score = score;
            }
        }
        const char* face_live_info[] = {"2d fake face","true face","2d fake face"};
        char face_live_txt[256];
        sprintf(face_live_txt, "%s %.2f", face_live_info[class_index], class_score);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(face_live_txt, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height*3 - baseLine*2;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height*2 + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, face_live_txt, cv::Point(x, y + label_size.height*2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  face live over ...");

    }


    //关键点
    {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  landmarks start ...");
        if(landmarks.size()!=0)
        {
            for(int i = 0; i <= landmarks.size()/2; i++)
            {
                //__android_log_print(ANDROID_LOG_DEBUG, "ncnn", "pfld detect.%f  %f  ...",landmarks[i*2],landmarks[i*2+1]);
                cv::circle(rgb, cv::Point(landmarks[i*2],landmarks[i*2+1]),2,cv::Scalar(0, 0, 255), -1);
            }
        }
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  landmarks over ...");
    }


    //画矫正后的人脸
    {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  standard_face start ...");
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  standard_face rgb:%i %i.%i",rgb.rows,rgb.cols,rgb.channels());
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  standard_face standard_face_cv :%i %i.%i",standard_face_cv.rows,standard_face_cv.cols,standard_face_cv.channels());
        cv::Rect roi_rect = cv::Rect(0, 0, standard_face_cv.cols, standard_face_cv.rows);
        cv::Mat standard_face_cv_8U(standard_face_cv.cols, standard_face_cv.rows, CV_8UC3);
        standard_face_cv.convertTo(standard_face_cv_8U, CV_8UC3);
        standard_face_cv_8U.copyTo(rgb(roi_rect));

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  standard_face over ...");
    }

    //人脸属性 人脸识别系统中并不需要
    {
//        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  attribute start ...");
//        const char* sex[] = {"man","female"};
//        const char* glass[] = {"no glass","glass"};
//        const char* mask[] = {"no mask","mask"};
//        const char* age[] = {"0-6","7-18","19-30","31-60","61+"};
//        const char* hat[] = {"no hat","hat"};
//        char attribute_txt[256];
//        sprintf(attribute_txt, "%s %s %s %s %s", sex[1], glass[1],mask[1],age[1],hat[1]);
//        int baseLine = 0;
//        cv::Size label_size = cv::getTextSize(attribute_txt, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//        int x = obj.rect.x;
//        int y = obj.rect.y - label_size.height*5 - baseLine;
//        if (y < 0)
//            y = 0;
//        if (x + label_size.width > rgb.cols)
//            x = rgb.cols - label_size.width;
//
//        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height*2 + baseLine)),
//                      cv::Scalar(255, 255, 255), -1);
//
//        cv::putText(rgb, attribute_txt, cv::Point(x, y + label_size.height*2),
//                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
//
//        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw  attribute over ...");
    }


    return 0;
}

int Utils::face_correction(cv::Mat& rgb, cv::Mat &standard_face_cv,std::vector<float> landmarks)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "face_correction  start ...");
    // 计算变换矩阵 并且求逆变换
    int type = 0;       // 0->区域外填充为v[0],v[1],v[2], -233->区域外不处理
    float tm[6];
    float tm_inv[6];
    float point_dst[10] = { // +8 是因为我们处理112*112的图
            30.2946f + 8.0f, 51.6963f,
            65.5318f + 8.0f, 51.5014f,
            48.0252f + 8.0f, 71.7366f,
            33.5493f + 8.0f, 92.3655f,
            62.7299f + 8.0f, 92.2041f,
    };
    // 人脸区域在原图上的坐标
    float point_src[10] = {
            static_cast<float>(landmarks[0]), static_cast<float>(landmarks[1]),
            static_cast<float>(landmarks[2]), static_cast<float>(landmarks[3]),
            static_cast<float>(landmarks[4]), static_cast<float>(landmarks[5]),
            static_cast<float>(landmarks[6]), static_cast<float>(landmarks[7]),
            static_cast<float>(landmarks[8]), static_cast<float>(landmarks[9]),
    };
    ncnn::get_affine_transform(point_src, point_dst, 5, tm);
    ncnn::invert_affine_transform(tm, tm_inv);

//    ncnn::Mat dst(112, 112, 3);
//    cv::Mat cv_img = cv::Mat::zeros(standard_face.w,standard_face.h,CV_8UC3);
    ncnn::warpaffine_bilinear_c3(static_cast<const unsigned char *>(rgb.data), rgb.cols, rgb.rows,
                                 static_cast<unsigned char *>(standard_face_cv.data), standard_face_cv.cols, standard_face_cv.rows,
                                 tm_inv);

//    standard_face.to_pixels(cv_img.data, ncnn::Mat::PIXEL_RGB);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "face_correction  over ...");
    return 0;

}

int Utils::get_max_index(std::vector<float> output, int start, int end) {

    return 1;
//    std::vector<float> tem_vector;
//    tem_vector.insert(tem_vector.end(), output.begin()+start,output.begin()+end);
//    float smallest = std::min_element(std::begin(output), std::end(output));
//    int index = std::distance(std::begin(output), smallest);
//    return index;

}


