//
// Created by mingren on 2020/11/22.
//

#ifndef MNN_DEV_ENV_CLASSIFY_H
#define MNN_DEV_ENV_CLASSIFY_H

#include <iostream>
#include <string>
#include "Defs.h"
#include "MNNDefine.h"
#include "MnnNet.hpp"
#include "opencv2/opencv.hpp"

#define CLS_IMG_C 3
#define CLS_IMG_W 128
#define CLS_IMG_H 128

enum ErrorCode {
    SUCCESS          = 0,
    MALLOC_ERROR     = 10,
    LABLE_PATH_ERROR = 11,
    BOX_SIZE_ERROR   = 12,
};

class ModelConfig {
public:
    std::string modelPath;
    std::string lablePath;
};

class InputParams {
public:
    cv::Mat                            img;        // 图像输入
    ccv::Box                           box;        // 图像需识别的boundingbox
    float                              threshold;  // 分类阈值
    std::map<std::string, ModelConfig> clsConfigs;
};

/**
 * @describe : 分类lable描述
 * @comments :
 *
 */
class Classifiers {
public:
    Classifiers(int id, std::string description) {
        this->id          = id;
        this->description = description;
    }

public:
    int         id;
    std::string description;
};

/**
 * @describe : 分类lable导入容器
 * @comments :
 *
 */
class ClsLable {
public:
    // 用于初始化细分类的类别
    std::set<int>                           uniqueTSRCode;
    std::map<std::string, std::vector<int>> positiveIndexDict;
    std::map<std::string, std::vector<int>> positiveTsrDict;
    std::map<std::string, std::set<int>>    classIdSet;
};

class ClsRes {
public:
    int   curTsr;
    int   curMaxClass;
    float curMaxClassScore;
    float curMaxClassLogit;
};

class Classify {
public:
    ~Classify() {}

    ErrorCode init(InputParams &params);
    ErrorCode infer();
    ErrorCode release();
    ClsRes    getRes() { return mRes; }

private:
    ErrorCode imgProc();
    ErrorCode process();
    ErrorCode initClsLabel(std::map<std::string, ModelConfig> &clsConfigs);

public:
    ClsLable mClsLable;  // 细分类的属性
private:
    InputParams  mInputParams;
    float *      mImgData;  // 细分类的图像数据
    ccv::MnnNet *mNet;      // 细分类的模型对象
    ClsRes       mRes;      // 存储当前粗分类的输出
};

#endif  // MNN_DEV_ENV_CLASSIFY_H
