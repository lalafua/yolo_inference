#include "../include/detector.hpp"
#include <vector>

Detector::Detector(const std::string& modelPath, bool useGPU) {
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11_Inference");
    sessionOptions = Ort::SessionOptions();

    if (useGPU) {
        // 如果安装了 CUDA 版 ORT，可以启用 GPU
        // OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    }

    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);

    // 获取输入/输出节点信息 (YOLOv11 默认通常是 1个输入 1个输出)
    Ort::AllocatorWithDefaultOptions allocator;
    inputName = session->GetInputNameAllocated(0, allocator).get();
    outputName = session->GetOutputNameAllocated(0, allocator).get();
    inputNodeDims = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

void Detector::preprocess(cv::Mat& image, float& scale, cv::Scalar& pad) {
    int w = image.cols;
    int h = image.rows;
    scale = std::min((float)modelShape.width / w, (float)modelShape.height / h);
    int nw = int(w * scale);
    int nh = int(h * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(nw, nh));

    int top = (modelShape.height - nh) / 2;
    int bottom = modelShape.height - nh - top;
    int left = (modelShape.width - nw) / 2;
    int right = modelShape.width - nw - left;
    pad = cv::Scalar(left, top);

    cv::copyMakeBorder(resized, image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
}

void Detector::setClassNames (const std::vector<std::string>& names) {
    this->classNames = names;
}

bool Detector::detect(cv::Mat& frame, std::vector<Detection>& output, float confThreshold, float iouThreshold) {
    cv::Mat blob;
    cv::cvtColor(frame, blob, cv::COLOR_BGR2RGB);
    
    float scale;
    cv::Scalar pad;
    preprocess(blob, scale, pad);

    // 归一化并转为 NCHW
    blob.convertTo(blob, CV_32FC3, 1.0 / 255.0);
    cv::Mat tensor = cv::dnn::blobFromImage(blob);

    // 创建输入 Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*)tensor.data, tensor.total(), inputNodeDims.data(), inputNodeDims.size());

    // 运行推理
    const char* inputNames[] = { inputName.c_str() };
    const char* outputNames[] = { outputName.c_str() };
    auto outputTensors = session->Run(Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, outputNames, 1);

    // 后处理
    float* rawOutput = outputTensors[0].GetTensorMutableData<float>();
    auto shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, 84, 8400]

    int rows = shape[1]; // 4 (box) + num_classes
    int anchors = shape[2]; // 8400

    // 转置数据处理：YOLOv11 输出是 [1, 4+N, 8400]
    cv::Mat outputMat(rows, anchors, CV_32F, rawOutput);
    outputMat = outputMat.t(); // 变成 [8400, 84]

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < anchors; i++) {
        cv::Mat row = outputMat.row(i);
        cv::Mat scores = row.colRange(4, rows);
        cv::Point classIdPoint;
        double maxConf;
        cv::minMaxLoc(scores, 0, &maxConf, 0, &classIdPoint);

        if (maxConf > confThreshold) {
            float cx = row.at<float>(0);
            float cy = row.at<float>(1);
            float w = row.at<float>(2);
            float h = row.at<float>(3);

            int left = static_cast<int>((cx - 0.5 * w - pad.val[0]) / scale);
            int top = static_cast<int>((cy - 0.5 * h - pad.val[1]) / scale);
            int width = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);

            classIds.push_back(classIdPoint.x);
            confidences.push_back((float)maxConf);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    // NMS 非极大值抑制
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.conf = confidences[idx];
        det.classId = classIds[idx];
        
        if (det.classId >= 0 && det.classId < (int)this->classNames.size()) {
            det.className = this->classNames[det.classId];
        } else {
            det.className = "ID: " + std::to_string(det.classId);
        }
        
        output.push_back(det);
    }

    return !output.empty();
}