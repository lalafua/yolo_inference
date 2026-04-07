#include "../include/gui.hpp"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    this->setWindowTitle("Inference");
    this->resize(1000, 700);

    // 1. 初始化推理引擎
    detector = std::make_unique<Detector>("/home/lalafua/Workspace/毕业设计/yolo-inference/models/best.onnx", false);
    detector->setClassNames({
        "black_core", "corner", "crack", "finger", "fragment", 
        "horizontal_dislocation", "printing_error", "scratch", 
        "short_circuit", "star_crack", "thick_line", "vertical_dislocation"
    });

    // 2. 布局 UI
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    btnUpload = new QPushButton("上传", this);
    btnUpload->setFixedHeight(40);
    
    imgDisplay = new QLabel("尚未加载图片", this);
    imgDisplay->setAlignment(Qt::AlignCenter);
    imgDisplay->setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0;");

    mainLayout->addWidget(btnUpload);
    mainLayout->addWidget(imgDisplay);
    setCentralWidget(centralWidget);

    // 3. 绑定事件
    connect(btnUpload, &QPushButton::clicked, this, &MainWindow::onUploadImage);
}

MainWindow::~MainWindow() {}

void MainWindow::onUploadImage() {
    QString filePath = QFileDialog::getOpenFileName(this, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    
    if (!filePath.isEmpty()) {
        // A. 读取图片
        cv::Mat frame = cv::imread(filePath.toStdString());
        if (frame.empty()) return;

        // B. 调用推理接口
        std::vector<Detection> results;
        detector->detect(frame, results);

        // C. 在 Mat 上绘制结果 (也可在 GUI 绘图层画，这里为了方便直接处理 Mat)
        drawResults(frame, results);

        // D. 显示到界面
        QImage qimg = cvMatToQImage(frame);
        QPixmap pixmap = QPixmap::fromImage(qimg);
        
        // 缩放图片以适应 Label 大小，保持比例
        imgDisplay->setPixmap(pixmap.scaled(imgDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void MainWindow::drawResults(cv::Mat& img, const std::vector<Detection>& results) {
    for (const auto& det : results) {
        // 绘制矩形框
        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
        
        // 绘制标签背景
        std::string label = det.className + " (" + std::to_string(det.conf).substr(0, 4) + ")";
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
        // cv::rectangle(img, cv::Rect(det.box.x, det.box.y - labelSize.height - 5, labelSize.width, labelSize.height + 5),cv::Scalar(0, 0, 0), cv::FILLED);
        
        // 绘制白色文字
        cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}

QImage MainWindow::cvMatToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC3) {
        // 转换 BGR 到 RGB
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage((const unsigned char*)(rgb.data), rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    }
    return QImage();
}