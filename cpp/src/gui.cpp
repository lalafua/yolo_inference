#include "../include/gui.hpp"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    this->setWindowTitle("YOLO Inference");
    this->resize(1200, 800);

    // 1. 初始化推理引擎 (保持不变)
    detector = std::make_unique<Detector>("/home/lalafua/Workspace/毕业设计/yolo-inference/models/best.onnx", false);
    detector->setClassNames({
        "black_core", "corner", "crack", "finger", "fragment", 
        "horizontal_dislocation", "printing_error", "scratch", 
        "short_circuit", "star_crack", "thick_line", "vertical_dislocation"
    });

    // 2. 布局 UI
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // --- 顶部按钮 ---
    btnUpload = new QPushButton("上传图片并识别", this);
    btnUpload->setFixedHeight(45);
    // btnUpload->setStyleSheet("font-size: 16px; font-weight: bold;");
    mainLayout->addWidget(btnUpload);

    // --- 中间显示区域 (水平排列左右两栏) ---
    QHBoxLayout *columnsLayout = new QHBoxLayout();

    // 样式定义
    QString titleStyle = "font-size: 14px; font-weight: bold; color: #333; margin-bottom: 5px;";
    QString imageBoxStyle = "border: 2px solid #bbb; background-color: #f9f9f9;";

    // [左栏：原始图像]
    QVBoxLayout *leftColumn = new QVBoxLayout();
    QLabel *leftTitle = new QLabel("原始图像", this);
    leftTitle->setAlignment(Qt::AlignCenter);
    //leftTitle->setStyleSheet(titleStyle);
    
    leftDisplay = new QLabel("等待上传...", this);
    leftDisplay->setAlignment(Qt::AlignCenter);
    leftDisplay->setStyleSheet(imageBoxStyle);
    leftDisplay->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding); // 自动伸展

    leftColumn->addWidget(leftTitle);
    leftColumn->addWidget(leftDisplay);

    // [右栏：识别结果]
    QVBoxLayout *rightColumn = new QVBoxLayout();
    QLabel *rightTitle = new QLabel("识别结果", this);
    rightTitle->setAlignment(Qt::AlignCenter);
    //rightTitle->setStyleSheet(titleStyle);

    rightDisplay = new QLabel("等待识别...", this);
    rightDisplay->setAlignment(Qt::AlignCenter);
    rightDisplay->setStyleSheet(imageBoxStyle);
    rightDisplay->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    rightColumn->addWidget(rightTitle);
    rightColumn->addWidget(rightDisplay);

    // 将左右两栏加入水平布局
    columnsLayout->addLayout(leftColumn);
    columnsLayout->addLayout(rightColumn);

    // 将水平布局加入主布局
    mainLayout->addLayout(columnsLayout);

    setCentralWidget(centralWidget);

    // 3. 绑定事件
    connect(btnUpload, &QPushButton::clicked, this, &MainWindow::onUploadImage);
}

MainWindow::~MainWindow() {}

void MainWindow::onUploadImage() {
    QString filePath = QFileDialog::getOpenFileName(this, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    
    if (!filePath.isEmpty()) {
        cv::Mat originalFrame = cv::imread(filePath.toStdString());
        if (originalFrame.empty()) return;

        // --- 处理左侧：原图 ---
        QImage qimgLeft = cvMatToQImage(originalFrame);
        // 使用 SmoothTransformation 保证缩放质量
        leftDisplay->setPixmap(QPixmap::fromImage(qimgLeft).scaled(
            leftDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

        // --- 处理右侧：识别结果 ---
        cv::Mat detectFrame = originalFrame.clone();
        std::vector<Detection> results;
        detector->detect(detectFrame, results);
        drawResults(detectFrame, results);

        QImage qimgRight = cvMatToQImage(detectFrame);
        rightDisplay->setPixmap(QPixmap::fromImage(qimgRight).scaled(
            rightDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void MainWindow::drawResults(cv::Mat& img, const std::vector<Detection>& results) {
    for (const auto& det : results) {
        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
        
        std::string label = det.className + " " + std::to_string(det.conf).substr(0, 4);
        cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}

QImage MainWindow::cvMatToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage((const unsigned char*)(rgb.data), rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    } else if (mat.type() == CV_8UC1) {
        return QImage((const unsigned char*)(mat.data), mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
    }
    return QImage();
}