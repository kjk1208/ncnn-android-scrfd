//#ifndef LSTR_H
//#define LSTR_H
//
//#include "ncnn/net.h"
//#include "YoloV5.h"
//
//
//class LSTR {
//public:
//    LSTR(AAssetManager *mgr, const char *param, const char *bin, bool useGPU);
//
//    ~LSTR();
//
//    std::vector<BoxInfo> detect(JNIEnv *env, jobject image, float threshold, float nms_threshold);
//    std::vector<std::string> labels{"golf-cart", "transover", "person", "tree", "t-box", "wastebasket", "stairs",
//                                    "stele", "drain", "bunker", "flag", "a-piece-of-wood", "lake", "sign", "tool",
//                                    "left", "right", "go-straight", "boat", "plastic-cylinder"};
//
//private:
//    static std::vector<BoxInfo>
//    decode_infer(ncnn::Mat &data, const yolocv::YoloSize &frame_size, int net_size, int num_classes, float threshold);
//
////    static void nms(std::vector<BoxInfo>& result,float nms_threshold);
//    ncnn::Net *Net;
//    int input_size_w = 800 ;
//    int input_size_h = 288 ;
//    int num_class = 80;
//public:
//    static LSTR *detector;
//    static bool hasGPU;
//    static bool toUseGPU;
//};
//
//
//#endif //LSTR_H
//
//// Tencent is pleased to support the open source community by making ncnn available.
////
//// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
////
//// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//// in compliance with the License. You may obtain a copy of the License at
////
//// https://opensource.org/licenses/BSD-3-Clause
////
//// Unless required by applicable law or agreed to in writing, software distributed
//// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//// CONDITIONS OF ANY KIND, either express or implied. See the License for the
//// specific language governing permissions and limitations under the License.

#ifndef SCRFD_H
#define SCRFD_H

#include <opencv2/core/core.hpp>

#include <net.h>

//struct FaceObject
//{
//    cv::Rect_<float> rect;
//    cv::Point2f landmark[5];
//    float prob;
//};

class LSTR
{
public:
    int load(const char* modeltype, bool use_gpu = false);
    int load(AAssetManager* mgr, const char* modeltype, bool use_gpu = false);
    int detect(const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, float prob_threshold = 0.5f, float nms_threshold = 0.45f);
    int draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects);

private:
    ncnn::Net lstr;
    bool has_kps;
    int input_size_w = 800 ;
    int input_size_h = 288 ;
};

#endif // SCRFD_H
