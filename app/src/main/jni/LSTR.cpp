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
int LSTR::detect(const cv::Mat& rgb, std::vector<v_Lane_point>& faceobjects, float prob_threshold, float nms_threshold)
{
    int width   = rgb.cols;
    int height  = rgb.rows;
    float lane_threshold = threshold;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, input_size_w, input_size_h);

    const float mean_vals[3]    = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    const float norm_vals[3]    = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = lstr.create_extractor();
    //ex.set_light_mode(true);
    //ex.set_num_threads(12);

    ex.input("input.1", in);

    ncnn::Mat out;

    ex.extract("200", out);

    //output을 anchor별로 position값 분류하기
    std::vector<v_Lane_point> result;

    auto result_point = decode_infer(out, lane_threshold);
    result.insert(result.begin(), result_point.begin(), result_point.end());
    return 0;
}

std::vector<v_Lane_point>LSTR::decode_infer(ncnn::Mat &data, float threshold) {
    std::vector<v_Lane_point> result;
    float col_sample_w = 8.070707071;
    //out_j = out_j[:, ::-1, :]
    //kjk 20211006 아래 3줄 추가
    const int griding_size  = NUM_GRIDDING_BOX;  //100
    const int row_num       = NUM_ANCHOR;             //40
    const int lane_cls_num  = NUM_LANE_CLASS;    //2
    const int LD_y_axis     = LD_POINT_Y;           //15    //lookahead distance y축 좌표        //LD 사용자 입력
    //kjk 20211006 위 3줄 추가
    float out[griding_size+1][row_num][lane_cls_num];
    float out1[row_num][lane_cls_num];
    float out2[row_num][lane_cls_num];

    float horizon_max = 0;
    int horizon_idx = griding_size;
    int channel = data.c;
    for (int y = 0; y < row_num; y++) {
        for (int l = 0; l < lane_cls_num; l++) {
            for(int x =0;x < griding_size+1 ;x ++) {
                ncnn::Mat c_data = data.channel(x);
                const float xdata = c_data.row(y)[l] ;//c_data[ 4 * y + l];
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
    for (int y =0;y < row_num ;y ++){
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
                    o *=(float)x;
                    if (x!=0) {
                        out2[y][l] += o;
                    }
                    else if (x == griding_size){

                    }
                    else {//==0
                        out2[y][l] = o;
                    }
                }
            }
            if(horizon_max_idx==griding_size){
                out2[y][l] = 0;
            }
        }
    };
    if (horizon_idx == 2011){
        //no result
    }else{
        for (int l =0;l < lane_cls_num ;l ++) {
            float sum = 0;
            for (int y = 0; y < row_num; y++) {
                sum += out2[ y][ l];
            }
            if (sum > 2) {
                for (int y = 0; y < row_num; y++) {
                    if (out2[y][l] > 0) {
                        if (y > 6) {
                            v_Lane_point lane;
                            if (y==LD_y_axis)
                                lane.label = 1002 + l;
                            else if (y == row_num - 1)
                                lane.label = 1004 + l;
                            else
                                lane.label = 1000 + l;
                            lane.score = out2[y][l];
                            float xx = (out2[y][l] * col_sample_w);
                            //kjk 20211006
                            //float yy = culane_row_anchor[y];
                            //kjk 20211006
                            float yy = tusimple_row_anchor[y];
                            lane.x = xx;
                            lane.y = yy;
                            result.push_back(lane);
                        }
                    }
                }
            }
        }
    }

    return result;
}

int LSTR::draw(cv::Mat& rgb, const std::vector<v_Lane_point>& v_lane)
{
    int width   = rgb.cols;
    int height  = rgb.rows;
    float lane_scaleX = rgb.cols / 800.f;
    float lane_scaleY = rgb.rows / 288.f;
    for (size_t i = 0; i < v_lane.size(); i++)
    {
        const v_Lane_point& position = v_lane[i];
        cv::circle(rgb,cv::Point(position.x, position.y),cv::Scalar(0,255,0));
    }

    return 0;
}
