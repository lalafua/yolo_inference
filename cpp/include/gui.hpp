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
#include <memory>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onUploadImage(); 

private:
    // UI 控件
    QPushButton *btnUpload;
    QLabel *leftDisplay;   // 原始图像
    QLabel *rightDisplay;  // 标注图像
    
    // 推理引擎
    std::unique_ptr<Detector> detector;

    QImage cvMatToQImage(const cv::Mat& mat);
    void drawResults(cv::Mat& img, const std::vector<Detection>& results);
};