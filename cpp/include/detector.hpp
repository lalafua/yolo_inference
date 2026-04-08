#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 检测结果结构体
struct Detection {
    cv::Rect box;
    float conf;
    int classId;
    std::string className;
};

class Detector {
public:
    Detector(const std::string& modelPath, bool useGPU = false);
    ~Detector() = default;

    void setClassNames(const std::vector<std::string>& names);
    // 推理接口
    bool detect(cv::Mat& frame, std::vector<Detection>& output, float confThreshold = 0.25f, float iouThreshold = 0.45f);

private:
    // 预处理
    void preprocess(cv::Mat& image, float& scale, cv::Scalar& pad);
    
    // 推理环境成员
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;

    // 模型信息
    std::vector<int64_t> inputNodeDims; // [1, 3, 640, 640]
    std::string inputName;
    std::string outputName;

    const cv::Size modelShape = cv::Size(640, 640);
    std::vector<std::string> classNames;
};