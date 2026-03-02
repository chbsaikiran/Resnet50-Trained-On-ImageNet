#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

static const float MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float STD[3]  = {0.229f, 0.224f, 0.225f};

static const std::map<std::string, int> WNID_TO_INDEX = {
    {"n02077923", 0},   // sea lion
    {"n02058221", 1},   // albatross
    {"n02051845", 2},   // pelican
    {"n02037110", 3},   // oystercatcher
    {"n02028035", 4},   // redshank
    {"n01440764", 5},   // tench
    {"n01443537", 6},   // goldfish
    {"n01484850", 7},   // great white shark
    {"n01491361", 8},   // tiger shark
    {"n01494475", 9},   // hammerhead
};

static const std::vector<std::string> CLASS_NAMES = {
    "sea lion",
    "albatross",
    "pelican",
    "oystercatcher",
    "redshank",
    "tench",
    "goldfish",
    "great white shark",
    "tiger shark",
    "hammerhead"
};

static const int NUM_CLASSES = 10;

static cv::Mat preprocess(const cv::Mat &frame)
{
    cv::Mat img;
    cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);

    int h = img.rows;
    int w = img.cols;

    int resize_size = 256;
    int crop_size   = 224;

    int new_h, new_w;
    if (h < w) {
        new_h = resize_size;
        new_w = static_cast<int>(w * resize_size / static_cast<float>(h));
    } else {
        new_w = resize_size;
        new_h = static_cast<int>(h * resize_size / static_cast<float>(w));
    }

    cv::resize(img, img, cv::Size(new_w, new_h));

    int start_x = (new_w - crop_size) / 2;
    int start_y = (new_h - crop_size) / 2;
    img = img(cv::Rect(start_x, start_y, crop_size, crop_size));

    img.convertTo(img, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    for (int i = 0; i < 3; i++)
        channels[i] = (channels[i] - MEAN[i]) / STD[i];
    cv::merge(channels, img);

    return img;
}

static void mat_to_chw(const cv::Mat &img, std::vector<float> &out)
{
    int channel_size = 224 * 224;
    out.resize(3 * channel_size);

    for (int c = 0; c < 3; c++)
        for (int i = 0; i < 224; i++)
            for (int j = 0; j < 224; j++)
                out[c * channel_size + i * 224 + j] =
                    img.at<cv::Vec3f>(i, j)[c];
}

struct ValSample {
    std::string path;
    int         label;
};

static std::vector<ValSample> collect_samples(const std::string &val_dir)
{
    std::vector<ValSample> samples;

    for (const auto &class_dir : fs::directory_iterator(val_dir)) {
        if (!class_dir.is_directory())
            continue;

        std::string wnid = class_dir.path().filename().string();

        auto it = WNID_TO_INDEX.find(wnid);
        if (it == WNID_TO_INDEX.end()) {
            std::cerr << "Warning: skipping unknown class folder " << wnid << "\n";
            continue;
        }
        int label = it->second;

        for (const auto &entry : fs::directory_iterator(class_dir)) {
            if (!entry.is_regular_file())
                continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                samples.push_back({entry.path().string(), label});
        }
    }

    std::sort(samples.begin(), samples.end(),
              [](const ValSample &a, const ValSample &b) {
                  return a.path < b.path;
              });

    return samples;
}

int main(int argc, char **argv)
{
    std::string model_path = "resnet18_imagenet10.onnx";
    std::string val_dir    = "imagenet10/val.X";

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) val_dir    = argv[2];

    /* ---- Load ONNX model -------------------------------------------- */

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet_accuracy");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(4);

    Ort::Session session(env, model_path.c_str(), session_options);

    auto input_names  = session.GetInputNames();
    auto output_names = session.GetOutputNames();
    const char *input_name  = input_names[0].c_str();
    const char *output_name = output_names[0].c_str();

    std::cout << "Model loaded: " << model_path << "\n";

    /* ---- Collect validation images ---------------------------------- */

    std::vector<ValSample> samples = collect_samples(val_dir);
    int total_images = static_cast<int>(samples.size());
    std::cout << "Validation images found: " << total_images << "\n\n";

    if (total_images == 0) {
        std::cerr << "No images found in " << val_dir << "\n";
        return 1;
    }

    /* ---- Inference loop --------------------------------------------- */

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> input_tensor_values;
    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    int correct = 0;
    int total   = 0;
    int per_class_correct[NUM_CLASSES] = {};
    int per_class_total[NUM_CLASSES]   = {};

    double total_inference_ms = 0.0;

    for (int idx = 0; idx < total_images; idx++) {
        const ValSample &sample = samples[idx];

        cv::Mat frame = cv::imread(sample.path, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "Warning: cannot read " << sample.path << "\n";
            continue;
        }

        cv::Mat img = preprocess(frame);
        mat_to_chw(img, input_tensor_values);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto t0 = std::chrono::high_resolution_clock::now();

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            &input_name,  &input_tensor,  1,
            &output_name, 1
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        total_inference_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        float *output = output_tensors[0].GetTensorMutableData<float>();
        int pred = static_cast<int>(
            std::max_element(output, output + NUM_CLASSES) - output);

        int gt = sample.label;
        per_class_total[gt]++;
        total++;

        if (pred == gt) {
            correct++;
            per_class_correct[gt]++;
        }

        if ((idx + 1) % 100 == 0 || idx == total_images - 1) {
            double running_acc = 100.0 * correct / total;
            std::printf("[%5d / %5d]  running accuracy: %.2f%%\n",
                        idx + 1, total_images, running_acc);
        }
    }

    /* ---- Results ---------------------------------------------------- */

    double accuracy = 100.0 * correct / total;

    std::cout << "\n====== Validation Results ======\n";
    std::cout << "Total images : " << total   << "\n";
    std::cout << "Correct      : " << correct << "\n";
    std::printf("Top-1 Accuracy: %.2f%%\n", accuracy);
    std::printf("Avg inference : %.3f ms\n", total_inference_ms / total);
    std::cout << "================================\n";

    std::cout << "\nPer-class accuracy:\n";
    for (int c = 0; c < NUM_CLASSES; c++) {
        if (per_class_total[c] > 0) {
            double cls_acc = 100.0 * per_class_correct[c] / per_class_total[c];
            std::printf("  [%d] %-20s  %4d / %4d  (%.2f%%)\n",
                        c, CLASS_NAMES[c].c_str(),
                        per_class_correct[c], per_class_total[c], cls_acc);
        }
    }

    return 0;
}

