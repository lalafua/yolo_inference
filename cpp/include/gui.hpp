#pragma once

#include "detector.hpp"
#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QPixmap>
#include <QImage>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onUploadImage(); // 响应上传按钮

private:
    // UI 控件
    QPushButton *btnUpload;
    QLabel *imgDisplay;
    
    // 推理引擎
    std::unique_ptr<Detector> detector;

    // 辅助函数：cv::Mat 转 QImage
    QImage cvMatToQImage(const cv::Mat& mat);
    // 绘图函数：在结果上绘制框和文字
    void drawResults(cv::Mat& img, const std::vector<Detection>& results);
};

