#include "LSTR.h"
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

bool LSTR::hasGPU = true;
bool LSTR::toUseGPU = true;
LSTR *LSTR::detector = nullptr;
std::vector<std::int16_t> culane_row_anchor = {121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287};
std::vector<std::int16_t> tusimple_row_anchor = {128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284};

#define NUM_GRIDDING_BOX 100
#define NUM_ANCHOR 40
#define NUM_LANE_CLASS 2
#define LD_POINT_Y 15

LSTR::LSTR(AAssetManager *mgr, const char *param, const char *bin, bool useGPU) {
//    hasGPU = ncnn::get_gpu_count() > 0;
//    toUseGPU = hasGPU && useGPU;
//
//    Net = new ncnn::Net();
//    ncnn::set_cpu_powersave(2);
//    ncnn::set_omp_num_threads(ncnn::get_cpu_count());
//    Net->opt = ncnn::Option();
//    // opt 需要在加载前设置
//#if NCNN_VULKAN
//    //Net->opt.use_vulkan_compute = hasGPU;
//#endif
//
//    //Net->opt.use_vulkan_compute = true;  // gpu
//    //Net->opt.use_winograd_convolution = true;
//    //Net->opt.use_sgemm_convolution = true;
//    //Net->opt.use_fp16_packed  = true;  // fp16运算加速
//    //Net->opt.use_fp16_storage  = true;  // fp16运算加速
//
//    //kjk221209
//    //Net->opt.use_fp16_arithmetic = true;  // fp16运算加速
//
//    //Net->opt.use_packing_layout = true;
//    //Net->opt.use_shader_pack8 = false;
//    //Net->opt.use_image_storage = false;
//    //Net->opt.use_int8_inference = true;
//    Net->load_param(mgr, param);
//    Net->load_model(mgr, bin);


    hasGPU = ncnn::get_gpu_count() > 0;
    toUseGPU = hasGPU && useGPU;

    Net = new ncnn::Net();
    // opt 需要在加载前设置

    Net->opt.use_vulkan_compute = true;  // gpu

    //Net->opt.use_winograd_convolution = true;
    //Net->opt.use_sgemm_convolution = true;
    //Net->opt.use_fp16_packed  = true;  // fp16运算加速
    //Net->opt.use_fp16_storage  = true;  // fp16运算加速
    Net->opt.use_fp16_arithmetic = true;  // fp16运算加速
    //Net->opt.use_packing_layout = true;
    //Net->opt.use_shader_pack8 = false;
    //Net->opt.use_image_storage = false;
    //Net->opt.use_int8_inference = true;
    Net->load_param(mgr, param);
    Net->load_model(mgr, bin);
}

LSTR::~LSTR() {
    Net->clear();
    delete Net;
}

std::vector<BoxInfo> LSTR::detect(JNIEnv *env, jobject image, float threshold, float nms_threshold) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);
    ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB, input_size_w,
                                                             input_size_h);
    //__android_log_print(ANDROID_LOG_VERBOSE,"jun","size width : %u , height : %u",img_size.width, img_size.height);
    float mean[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};//{1 / 255.f, 1 / 255.f, 1 / 255.f};
    float norm[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    in_net.substract_mean_normalize(mean, norm);
    /*float test_color1 = in_net.channel(0).row(1)[1];
    float test_color2 = in_net.channel(1).row(1)[1];
    float test_color3 = in_net.channel(2).row(1)[1];
    float t=test_color1+test_color2+test_color3;
*/
    ncnn::Extractor ex = Net->create_extractor();
    ex.set_light_mode(true);            //blob memory 할당후 해제되어 메모리 사용량 증가
    ex.set_num_threads(12);
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
    ex.input(0, in_net);
    std::vector<BoxInfo> result;
    ncnn::Mat out;
    ex.extract("200", out);

    auto boxes = decode_infer(out, {(int) img_size.width, (int) img_size.height}, input_size_w, num_class, threshold);
    result.insert(result.begin(), boxes.begin(), boxes.end());

/////////////////////////////    nms(result,nms_threshold);
    return result;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector<BoxInfo>
LSTR::decode_infer(ncnn::Mat &data, const yolocv::YoloSize &frame_size, int net_size, int num_classes, float threshold) {
    std::vector<BoxInfo> result;
    //data 201,18,4  200 x轴 ，18 Y轴，4车道
    float col_sample_w = 8.070707071;
    //out_j = out_j[:, ::-1, :]
    //kjk 20211006 아래 3줄 추가
    const int griding_size = NUM_GRIDDING_BOX;  //100
    const int row_num = NUM_ANCHOR;             //40
    const int lane_cls_num = NUM_LANE_CLASS;    //2
    const int LD_y_axis = LD_POINT_Y;           //15    //lookahead distance y축 좌표        //LD 사용자 입력
    //kjk 20211006 위 3줄 추가
    float out[griding_size+1][row_num][lane_cls_num];
    float out1[row_num][lane_cls_num];
    float out2[row_num][lane_cls_num];

    //kjk 20211006
    //float out[201][18][4];
    //float out1[18][4];
    //float out2[18][4];
    //kjk 20211006

    //#第二个纬度 倒序
    //print("out_j.shape 1",out_j.shape)
    //沿着Z 轴 进行softmax ，每个数 乘以 【1~200]  代表着 图像X 定位的位置。
    //比如 下标 1 ，数值0.9 ，乘以 1 = X分割区域点 1 的位置概率是 0.9
    //下标100 ，数值 0.8，乘以 100 = 分割区域点 100 处，出现概率是 0.8
    //车道最终预测结果取最大，类似一个长的山峰，沿着最高点，选择高处的连线
    //prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    //idx = np.arange(200) + 1
    //idx = idx.reshape(-1, 1, 1)
    //loc = np.sum(prob * idx, axis=0)
    float horizon_max = 0;

    int horizon_idx = griding_size;

    //kjk 20211006
    //int horizon_idx = 200;
    //kjk 20211006

    //float sum_exp[200];
    int channel = data.c;

    //kjk 20211006
//    for (int y = 0; y < 18; y++) {
//        for (int l = 0; l < 4; l++) {
//            for(int x =0;x < 201 ;x ++) {
    //kjk 20211006
    for (int y = 0; y < row_num; y++) {
        for (int l = 0; l < lane_cls_num; l++) {
            for(int x =0;x < griding_size+1 ;x ++) {

                ncnn::Mat c_data = data.channel(x);
                const float xdata = c_data.row(y)[l] ;//c_data[ 4 * y + l];

                //float xdata = data.channel(x).row(y)[l];
                //__android_log_print(ANDROID_LOG_VERBOSE,"jun","original_data[%d][%d][%d] dims : %f",x,y,l,xdata);;
                //__android_log_print(ANDROID_LOG_VERBOSE,"jun","my_data[%d][%d][%d] dims : %f",x,y,l,xx_data);;

                //int idx = l+ 18*y + 201 * x;
                //float expp = expf(xdata);
                float expp = expf(xdata);
                out[x][y][l] = expp;
                if (x!=0) {
                    out1[y][l] += expp;
                }
                else if (x == griding_size){

                }
                else {//==0
                    out1[y][l] = expp;
                }
            }
        }
    }

//kjk1020 input 찍어보기
//    for(int x =0;x < griding_size+1 ;x ++) {
//        __android_log_print(ANDROID_LOG_VERBOSE,"jun","original_data[%d][0][0] value : %f",x,out[x][0][0]);
//    }

//    for (int y =0;y < 18 ;y ++){
//        //const float *row_data= data.row(y);
//        for (int l =0;l < 4 ;l ++){
    for (int y =0;y < row_num ;y ++){
        //const float *row_data= data.row(y);
        for (int l =0;l < lane_cls_num ;l ++){
            float horizon_sum = 0;
            float horizon_max = 0;
            int horizon_max_idx = 0;
            for(int x =0;x < griding_size+1 ;x ++) {
                if (out1[y][l]!=0){
                    float o = out[x][y][l];
                    o /= out1[y][l] ;
                    if(o>horizon_max){
                        horizon_max = o;
                        horizon_max_idx = x;
                    }
                    //out2 = np.sum(prob * idx, axis=0)
                    o *=(float)x;
                    //out[x][y][l] = o;
                    /*
                     out[x][y][l] /= sum_exp ;
                    out[x][y][l] *=(float)x;
                     */
                    if (x!=0) {
                        out2[y][l] += o;
                    }
                    else if (x == griding_size){

                    }
                    else {//==0
                        out2[y][l] = o;
                    }
                    //horizon_sum +=o;
                }
            }
            //kjk 20211006
//            if(horizon_max_idx==200){
//                out2[y][l] = 0;
//            }
            //kjk 20211006

            if(horizon_max_idx==griding_size){
                out2[y][l] = 0;
            }

            /*sum_exp[x] = horizon_sum;
            if(horizon_sum > horizon_max){
                horizon_idx = x;
                horizon_max = horizon_sum;
            }*/
        }
    };
    //out_j = np.argmax(out_j, axis=0)

    //loc[out_j == cfg.griding_num] = 0
    if (horizon_idx == 2011){
        //no result

    }else{
        //#out_j (18,4) ,4 条车道，存储x 的位置[0~1]，18 是Y 的序号
        //for i in range(out_j.shape[1]):

//kjk 20211006
//        for (int l =0;l < 4 ;l ++) {
//            //#10% 左侧区域开始
//            //if np.sum(out_j[:, i] != 0) > 1:
//            float sum = 0;
//            for (int y = 0; y < 18; y++) {
//                //const float *row_data = data.row(y);
//                sum += out2[ y][ l];
//            }
//            if (sum > 2) {
//
//                //for k in range(out_j.shape[0]):
//                //if out_j[k, i] > 0:
//                for (int y = 0; y < 18; y++) {

//kjk 20211006
        for (int l =0;l < lane_cls_num ;l ++) {
            //#10% 左侧区域开始
            //if np.sum(out_j[:, i] != 0) > 1:
            float sum = 0;
            for (int y = 0; y < row_num; y++) {
                //const float *row_data = data.row(y);
                sum += out2[ y][ l];
            }
            if (sum > 2) {
                //for k in range(out_j.shape[0]):
                //if out_j[k, i] > 0:
                for (int y = 0; y < row_num; y++) {
                    if (out2[y][l] > 0) {
                        if (y > 6) {
                            BoxInfo box;
                            //ppp = (int(out_j[k, i] * col_sample_w ) - 1,
                            //int((row_anchor[cls_num_per_lane-1-k])) - 1 )
                            if (y==LD_y_axis)
                                box.label = 1002 + l;
                            else if (y == row_num - 1)
                                box.label = 1004 + l;
                            else
                                box.label = 1000 + l;
                            box.score = out2[y][l];
                            float xx = (out2[y][l] * col_sample_w);
                            //kjk 20211006
                            //float yy = culane_row_anchor[y];
                            //kjk 20211006
                            float yy = tusimple_row_anchor[y];
                            box.x1 = xx;
                            box.y1 = yy;
                            box.x2 = xx + 1;
                            box.y2 = yy + 1;
                            result.push_back(box);
                        }
                    }
                }
            }
        }
    }

    return result;
}

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "LSTR.h"

#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

std::vector<std::int16_t> culane_row_anchor = {121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287};
std::vector<std::int16_t> tusimple_row_anchor = {128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284};

#define NUM_GRIDDING_BOX 100
#define NUM_ANCHOR 40
#define NUM_LANE_CLASS 2
#define LD_POINT_Y 15

int LSTR::load(const char* modeltype, bool use_gpu)
{
    lstr.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    lstr.opt = ncnn::Option();

#if NCNN_VULKAN
    lstr.opt.use_vulkan_compute = use_gpu;
#endif

    lstr.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "ufast_lane_det-sim-opt-fp16_sunsan_background.param"/*, modeltype*/);
    sprintf(modelpath, "ufast_lane_det-sim-opt-fp16_sunsan_background.bin"/*, modeltype*/);

    lstr.load_param(parampath);
    lstr.load_model(modelpath);

    has_kps = strstr(modeltype, "_kps") != NULL;

    return 0;
}

int LSTR::load(AAssetManager* mgr, const char* modeltype, bool use_gpu)
{
    lstr.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    lstr.opt = ncnn::Option();


    lstr.opt.use_vulkan_compute = use_gpu;


    lstr.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "ufast_lane_det-sim-opt-fp16_sunsan_background.param"/*, modeltype*/);
    sprintf(modelpath, "ufast_lane_det-sim-opt-fp16_sunsan_background.bin"/*, modeltype*/);


    lstr.load_param(mgr, parampath);
    lstr.load_model(mgr, modelpath);

    has_kps = strstr(modeltype, "_kps") != NULL;

    return 0;
}

//vector 변경
int LSTR::detect(const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // insightface/detection/scrfd/configs/scrfd/scrfd_500m.py
    const int target_size_w = input_size_w;
    const int target_size_h = input_size_h;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1/128.f, 1/128.f, 1/128.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = scrfd.create_extractor();

    ex.input("input.1", in_pad);

    std::vector<FaceObject> faceproposals;

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_8", score_blob);
        ex.extract("bbox_8", bbox_blob);
        if (has_kps)
            ex.extract("kps_8", kps_blob);

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_16", score_blob);
        ex.extract("bbox_16", bbox_blob);
        if (has_kps)
            ex.extract("kps_16", kps_blob);

        const int base_size = 64;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }

    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_32", score_blob);
        ex.extract("bbox_32", bbox_blob);
        if (has_kps)
            ex.extract("kps_32", kps_blob);

        const int base_size = 256;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects8);

        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(faceproposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    int face_count = picked.size();

    faceobjects.resize(face_count);
    for (int i = 0; i < face_count; i++)
    {
        faceobjects[i] = faceproposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (faceobjects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (faceobjects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (faceobjects[i].rect.x + faceobjects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (faceobjects[i].rect.y + faceobjects[i].rect.height - (hpad / 2)) / scale;

        x0 = std::max(std::min(x0, (float)width - 1), 0.f);
        y0 = std::max(std::min(y0, (float)height - 1), 0.f);
        x1 = std::max(std::min(x1, (float)width - 1), 0.f);
        y1 = std::max(std::min(y1, (float)height - 1), 0.f);

        faceobjects[i].rect.x = x0;
        faceobjects[i].rect.y = y0;
        faceobjects[i].rect.width = x1 - x0;
        faceobjects[i].rect.height = y1 - y0;

        if (has_kps)
        {
            float x0 = (faceobjects[i].landmark[0].x - (wpad / 2)) / scale;
            float y0 = (faceobjects[i].landmark[0].y - (hpad / 2)) / scale;
            float x1 = (faceobjects[i].landmark[1].x - (wpad / 2)) / scale;
            float y1 = (faceobjects[i].landmark[1].y - (hpad / 2)) / scale;
            float x2 = (faceobjects[i].landmark[2].x - (wpad / 2)) / scale;
            float y2 = (faceobjects[i].landmark[2].y - (hpad / 2)) / scale;
            float x3 = (faceobjects[i].landmark[3].x - (wpad / 2)) / scale;
            float y3 = (faceobjects[i].landmark[3].y - (hpad / 2)) / scale;
            float x4 = (faceobjects[i].landmark[4].x - (wpad / 2)) / scale;
            float y4 = (faceobjects[i].landmark[4].y - (hpad / 2)) / scale;

            faceobjects[i].landmark[0].x = std::max(std::min(x0, (float)width - 1), 0.f);
            faceobjects[i].landmark[0].y = std::max(std::min(y0, (float)height - 1), 0.f);
            faceobjects[i].landmark[1].x = std::max(std::min(x1, (float)width - 1), 0.f);
            faceobjects[i].landmark[1].y = std::max(std::min(y1, (float)height - 1), 0.f);
            faceobjects[i].landmark[2].x = std::max(std::min(x2, (float)width - 1), 0.f);
            faceobjects[i].landmark[2].y = std::max(std::min(y2, (float)height - 1), 0.f);
            faceobjects[i].landmark[3].x = std::max(std::min(x3, (float)width - 1), 0.f);
            faceobjects[i].landmark[3].y = std::max(std::min(y3, (float)height - 1), 0.f);
            faceobjects[i].landmark[4].x = std::max(std::min(x4, (float)width - 1), 0.f);
            faceobjects[i].landmark[4].y = std::max(std::min(y4, (float)height - 1), 0.f);
        }
    }

    return 0;
}

int SCRFD::draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects)
{
    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];

//         fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(rgb, obj.rect, cv::Scalar(0, 255, 0));

        if (has_kps)
        {
            cv::circle(rgb, obj.landmark[0], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[1], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[2], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[3], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[4], 2, cv::Scalar(255, 255, 0), -1);
        }

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    return 0;
}
