//
// Created by mingren on 2020/11/22.
//

#include "Classify.h"

ErrorCode Classify::init(InputParams &params) {
    mInputParams = params;
    mImgData     = ((float *)std::malloc(CLS_IMG_C * CLS_IMG_W * CLS_IMG_H * sizeof(float)));
    if (mImgData == nullptr) {
        MNN_PRINT("ClassifyProcess::initEnv error!");
        return MALLOC_ERROR;
    }
    initClsLabel(params.clsConfigs);
    mNet = new ccv::MnnNet();
    return SUCCESS;
}

ErrorCode imgResize(cv::Mat &cur_img, ccv::Box &box, float *im_dst, int im_dst_w, int im_dst_h) {
    int cutxmin = box.xmin;
    int cutymin = box.ymin;
    int cutxmax = box.xmax;
    int cutymax = box.ymax;
    cutxmin     = cutxmin > 0 ? cutxmin : 0;
    cutymin     = cutymin > 0 ? cutymin : 0;
    cutxmax     = cutxmax <= cur_img.cols ? cutxmax : (cur_img.cols - 1);
    cutymax     = cutymax <= cur_img.rows ? cutymax : (cur_img.rows - 1);

    cv::Mat rgb_image;
    cv::cvtColor(cur_img, rgb_image, cv::COLOR_BGR2RGB);
    cv::Rect rect(cutxmin, cutymin, cutxmax - cutxmin, cutymax - cutymin);
    // 检测输出的bounding box 经过处理可能长宽为0，此情况不处理
    if ((cutxmax - cutxmin == 0) || (cutymax - cutymin) == 0) {
        MNN_PRINT("Bounding box width = %d, height = %d", cutxmax - cutxmin, cutymax - cutymin);
        return BOX_SIZE_ERROR;
    }

    cv::Mat input_im_patch = rgb_image(rect);
    cv::Mat outImage;
    outImage.create(im_dst_w, im_dst_h, CV_8UC3);
    cv::Size dsize(im_dst_w, im_dst_h);
    cv::resize(input_im_patch, outImage, dsize);

    int      c       = outImage.channels();
    int      w       = outImage.rows;
    int      h       = outImage.cols;
    uint8_t *outdata = (uint8_t *)outImage.data;
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int k = 0; k < h; ++k) {
                im_dst[i * w * h + j * h + k] = (outdata[c * k + c * h * j + i] / 255.0f - 0.5f) * 2.0f;
            }
        }
    }
    return SUCCESS;
}

ErrorCode Classify::imgProc() {
    ErrorCode ret = imgResize(mInputParams.img, mInputParams.box, mImgData, CLS_IMG_W, CLS_IMG_H);
    if (ret != SUCCESS) {
        return ret;
    }
    return SUCCESS;
}

int getMaxIndex(std::vector<float> vec) {
    if (vec.size() == 0) {
        return -1;
    }
    int maxIndex = 0;
    for (int i = 0; i < vec.size() - 1; i++) {
        if (vec[maxIndex] < vec[i + 1]) {
            maxIndex = i + 1;
        }
    }
    return maxIndex;
}

ErrorCode Classify::process() {
    float            max_cls_score = -1;
    std::vector<int> cur_positive_index;
    std::vector<int> cur_positive_tsr;

    for (auto clsConfig : mInputParams.clsConfigs) {
        std::string class_name = clsConfig.first;

        ccv::NetConfig cfg;
        cfg.modelPath = clsConfig.second.modelPath;
        cfg.saveTensorNames.push_back("MobilenetV2/Logits/Squeeze");
        cfg.outTensorNames.push_back("MobilenetV2/Logits/Squeeze");
        cfg.outTensorNames.push_back("MobilenetV2/Predictions/Softmax");

        mNet->init(cfg);

        auto        clsInput = mNet->input();
        MNN::Tensor givenTensor(clsInput, MNN::Tensor::CAFFE);
        const int   inputSize = givenTensor.elementSize();
        auto        inputData = givenTensor.host<float>();
        for (int i = 0; i < inputSize; ++i) {
            inputData[i] = static_cast<float>(mImgData[i]);
        }
        clsInput->copyFromHostTensor(&givenTensor);

        mNet->infer();

        auto   scoreHost = mNet->outputs("MobilenetV2/Predictions/Softmax");
        float *pscore    = (float *)std::malloc(scoreHost->elementSize() * sizeof(float));
        std::memcpy((void *)pscore, (const void *)scoreHost->host<float>(), scoreHost->elementSize() * sizeof(float));

        auto   squeezeHost = mNet->outputs("MobilenetV2/Logits/Squeeze");
        float *psqueeze    = (float *)std::malloc(squeezeHost->elementSize() * sizeof(float));
        std::memcpy((void *)psqueeze, (const void *)squeezeHost->host<float>(), squeezeHost->elementSize() * sizeof(float));

        cur_positive_index = mClsLable.positiveIndexDict[class_name];
        cur_positive_tsr   = mClsLable.positiveTsrDict[class_name];
        std::vector<float> positive_cur_class_pred;
        std::vector<float> positive_cur_class_logit;
        for (int i = 0; i < cur_positive_index.size(); ++i) {
            positive_cur_class_pred.push_back(pscore[cur_positive_index[i]]);
            positive_cur_class_logit.push_back(psqueeze[cur_positive_index[i]]);
        }
        int   cur_meta_max_cls_temp       = getMaxIndex(positive_cur_class_pred);
        float cur_meta_max_cls_score_temp = positive_cur_class_pred[cur_meta_max_cls_temp];

        if (cur_meta_max_cls_score_temp > max_cls_score) {
            max_cls_score         = cur_meta_max_cls_score_temp;
            mRes.curMaxClass      = cur_meta_max_cls_temp;
            mRes.curMaxClassScore = cur_meta_max_cls_score_temp;
            mRes.curMaxClassLogit = positive_cur_class_logit[mRes.curMaxClass];
        }
        if (pscore) {
            free(pscore);
            pscore = nullptr;
        }
        if (psqueeze) {
            free(psqueeze);
            psqueeze = nullptr;
        }
    }
    if (mRes.curMaxClassScore >= mInputParams.threshold) {
        mRes.curTsr = cur_positive_tsr[mRes.curMaxClass];
    } else {
        mRes.curTsr = -1;
    }

    return SUCCESS;
}

void split(const std::string &s, std::vector<std::string> &sv) {
    std::string v;
    char        emp = ' ';
    int         idx = 0;
    for (auto c : s) {
        idx++;
        if (c != emp) {
            v.push_back(c);
        }
        if ((c == emp || idx == s.size()) && !v.empty()) {
            sv.push_back(v);
            v.clear();
        }
    }
}

ErrorCode Classify::initClsLabel(std::map<std::string, ModelConfig> &clsConfigs) {
    // map< 细分类, vector<tsrcode, description>>
    std::map<std::string, std::vector<Classifiers>> class_label_dict;
    // map< 细分类, vector<tsrcode>>
    mClsLable.classIdSet.clear();
    // vector<tsrcode> 存储所有的tsr_code
    mClsLable.uniqueTSRCode.clear();
    // 将-1和471015的去除
    mClsLable.positiveIndexDict.clear();
    mClsLable.positiveTsrDict.clear();

    for (auto clsConfig : clsConfigs) {
        std::string   class_name     = clsConfig.first;
        std::string   lable_abs_path = clsConfig.second.lablePath;
        std::ifstream classifierFile;
        classifierFile.open(lable_abs_path);
        if (!classifierFile) {
            MNN_ERROR("分类lable的配置文件打开失败，请检查文件状态!");
            return LABLE_PATH_ERROR;
        }

        std::string line;
        while (getline(classifierFile, line)) {
            std::vector<std::string> sv;
            split(line, sv);
            if (sv[1] == "-1" && sv.size() == 2) {
                Classifiers classifiers1(atoi(sv[1].c_str()), "noise");
                class_label_dict[class_name].push_back(classifiers1);
            } else {
                Classifiers classifiers1(atoi(sv[1].c_str()), sv[2]);
                class_label_dict[class_name].push_back(classifiers1);
            }
            mClsLable.uniqueTSRCode.insert(atoi(sv[1].c_str()));
            mClsLable.classIdSet[class_name].insert(atoi(sv[1].c_str()));
        }
        classifierFile.close();
    }
    for (auto it = class_label_dict.begin(); it != class_label_dict.end(); it++) {
        std::string              key              = it->first;
        std::vector<Classifiers> classifiers_list = class_label_dict[key];
        std::vector<int>         positive_index_list;
        std::vector<int>         positive_tsr_list;

        for (int i = 0; i < classifiers_list.size(); ++i) {
            if (classifiers_list[i].id != -1 && classifiers_list[i].id != 471015) {
                positive_index_list.push_back(i);
                positive_tsr_list.push_back(classifiers_list[i].id);
            }
        }
        mClsLable.positiveIndexDict[key].insert(mClsLable.positiveIndexDict[key].end(), positive_index_list.begin(), positive_index_list.end());
        mClsLable.positiveTsrDict[key].insert(mClsLable.positiveTsrDict[key].end(), positive_tsr_list.begin(), positive_tsr_list.end());
    }
    return SUCCESS;
}

ErrorCode Classify::infer() {
    ErrorCode ret;
    ret = imgProc();
    if (ret != SUCCESS) {
        return ret;
    }
    ret = process();
    if (ret != SUCCESS) {
        return ret;
    }
    return SUCCESS;
}

ErrorCode Classify::release() {
    if (mNet) {
        mNet->release();
        mNet = nullptr;
    }
    if (mImgData) {
        free(mImgData);
        mImgData = nullptr;
    }
    return SUCCESS;
}
