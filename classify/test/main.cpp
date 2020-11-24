/* ***************************************************
 * Copyright © 2020 AutoNavi Software Co.,Ltd. All rights reserved.
 * File:    main.cpp
 * Purpose: for unite test
 * Author:  mingren (mingren.ms@autonavi.com)
 * Version: 1.0
 * Date:    2020/11/18
 * Update:
 *****************************************************/

#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include "Classify.h"

/**
 * 单模型测试case
 */
TEST(Classify, SingleInference)
{
    // 参数配置
    std::string img_path = "/Users/mingren/Documents/02.MNNHome/mnn_cdev_env/classify/images/1574493788617_ebfc5eb7-2fe4-4f96-8e2f-dace58120810.webp";

    ModelConfig modelCfg;
    modelCfg.modelPath = "/Users/mingren/Documents/02.MNNHome/mnn_cdev_env/classify/model/classifiers_models/rectangle_quant.mnn";
    modelCfg.lablePath = "/Users/mingren/Documents/02.MNNHome/mnn_cdev_env/classify/model/classifiers_lables/rectangle.txt";

    InputParams inputParams;
    inputParams.img = cv::imread(img_path);
    inputParams.box = {480.0,221.0,534.0,264.0, 0.0, 1};
    inputParams.threshold = 0.34;
    inputParams.clsConfigs.insert(std::make_pair("rectangle", modelCfg));

    // 模型inference及结果获取
    std::unique_ptr<Classify> cls_proc(new Classify);
    cls_proc->init(inputParams);
    cls_proc->infer();
    cls_proc->release();
    ClsRes curRes = cls_proc->getRes();
    EXPECT_EQ(curRes.curTsr, 400039);
}
/**
 *  任务多模型测试case
 *      1. 以任务为单位，分类模型串行运算
 *      2. input :
 */
TEST(Classify, TaskInference){



}

/**
 * 检测结果的接口？？
 */
class Detection{
public:
  // in
  cv::Mat img;
  std::vector<int> detClasses;
  std::vector<float> detScores;
  std::vector<ccv::Box> detBoxes;

  // out
  std::vector<int>   classifyTsr;
  std::vector<float> classifyScores;
  std::vector<float> classifyLogits;
  std::vector<int>   nonNoiseTsr;
};

int main(int argc, char *argv[])
{
    // 将main中的参数传递到google-test
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}