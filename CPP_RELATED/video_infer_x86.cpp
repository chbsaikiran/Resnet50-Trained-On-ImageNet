#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std;

const float MEAN[3] = {0.485f, 0.456f, 0.406f};
const float STD[3]  = {0.229f, 0.224f, 0.225f};
std::vector<std::string> class_names = {
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

cv::Mat preprocess(const cv::Mat& frame)
{
    cv::Mat img;
    cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);

    int h = img.rows;
    int w = img.cols;

    int resize_size = 256;
    int crop_size = 224;

    int new_h, new_w;

    if (h < w) {
        new_h = resize_size;
        new_w = int(w * resize_size / float(h));
    } else {
        new_w = resize_size;
        new_h = int(h * resize_size / float(w));
    }

    cv::resize(img, img, cv::Size(new_w, new_h));

    int start_x = (new_w - crop_size) / 2;
    int start_y = (new_h - crop_size) / 2;

    img = img(cv::Rect(start_x, start_y, crop_size, crop_size));

    img.convertTo(img, CV_32F, 1.0 / 255.0);

    vector<cv::Mat> channels(3);
    cv::split(img, channels);

    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - MEAN[i]) / STD[i];
    }

    cv::merge(channels, img);

    return img;
}

int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_options.SetIntraOpNumThreads(4);

    Ort::Session session(env, "./../resnet18_imagenet10.onnx", session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    //const char* input_name = session.GetInputName(0, allocator);
    //const char* output_name = session.GetOutputName(0, allocator);

    // Get input and output names (NEW API for ONNX Runtime >= 1.14)

    auto input_names = session.GetInputNames();
    auto output_names = session.GetOutputNames();

    const char* input_name = input_names[0].c_str();
    const char* output_name = output_names[0].c_str();

    cv::VideoCapture cap("./../imagenet10_val_video.mp4");

    while (cap.isOpened())
    {
        cv::Mat frame;
        if (!cap.read(frame))
            break;

        cv::Mat img = preprocess(frame);

        vector<float> input_tensor_values(1 * 3 * 224 * 224);

        int channel_size = 224 * 224;

        for (int c = 0; c < 3; c++)
            for (int i = 0; i < 224; i++)
                for (int j = 0; j < 224; j++)
                    input_tensor_values[c * channel_size + i * 224 + j] =
                        img.at<cv::Vec3f>(i, j)[c];

        vector<int64_t> input_shape = {1, 3, 224, 224};

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto start = chrono::high_resolution_clock::now();

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            &input_name,
            &input_tensor,
            1,
            &output_name,
            1
        );

        auto end = chrono::high_resolution_clock::now();

        double inference_time =
            chrono::duration<double, milli>(end - start).count();

        float* output =
            output_tensors[0].GetTensorMutableData<float>();

        int pred =
            max_element(output, output + 10) - output;

        std::string label = class_names[pred];

        char text[100];
        sprintf(text, "%s", label.c_str());

        cv::putText(frame, text,
                    cv::Point(30, 40),               // position
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,                             // font scale
                    cv::Scalar(0, 255, 0),           // green color
                    2);                              // thickness

        cout << "Prediction: " << label.c_str()
             << " | Time: " << inference_time
             << " ms\n";

        cv::imshow("x86 Inference", frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    return 0;
}