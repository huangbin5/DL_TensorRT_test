﻿#include <iostream>
#include <fstream>
// #include <vector>
#include <string>
#include <chrono>
#include <unordered_map>

#include "../private/tools.hpp"
#include "test_segment.hpp"

#if defined(_MSC_VER)
    #define IS_WINDOWS 1
#else
    #define IS_WINDOWS 0
#endif


int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    // const CfgType config = {
    //     {"model_path", std::string("/home/oyefish/data/eu_box_exp/exp03_0403_lvl/train6/weights/best.engine")},
    //     {"classes", std::vector<std::string>{"eu_box"}},
    //     {"model_w", 1280},
    //     {"model_h", 1280},
    //     {"conf", 0.25f},
    //     {"iou", 0.7f},
    // };
    auto config = Tools::parse_json_config("../config/segment.json");
    if constexpr (IS_WINDOWS) {
        config["model_path"] = std::string("D:/AI/data/0324_test/best.engine");
    }
    SegTest model(config,
                  IS_WINDOWS ? "D:/AI/data/0324_test" : "/home/oyefish/data/eu_box/0324_test",
                  true, true, true, false);
    const auto start = std::chrono::high_resolution_clock::now();
    model.infer_batch();
    // model.infer_single("Image_2025-03-24_14_52_34.png");
    const auto end = std::chrono::high_resolution_clock::now();
    std::cout << "一共耗时 " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << 's' << std::endl;
    return 0;
}
