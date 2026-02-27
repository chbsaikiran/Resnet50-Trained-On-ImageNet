/*
 * arm_wrapper.cpp
 *
 * ARM-side host application that runs on the Snapdragon applications
 * processor.  It captures video frames, preprocesses them with OpenCV,
 * sends the tensor to the Hexagon DSP via FastRPC for inference, and
 * displays the result.
 *
 * Build with the Android NDK (or Qualcomm ARM toolchain) and link
 * against the FastRPC stub library generated from resnet_infer.idl.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "resnet_infer.h"

/* ------------------------------------------------------------------ */
/*  rpcmem helpers – use ION/dma-buf for zero-copy FastRPC transfers   */
/* ------------------------------------------------------------------ */
#include "rpcmem.h"     /* from Hexagon SDK */

#define ION_HEAP_ID_SYSTEM  25

static uint8_t *alloc_rpcmem(int size)
{
    return (uint8_t *)rpcmem_alloc(ION_HEAP_ID_SYSTEM,
                                   RPCMEM_DEFAULT_FLAGS,
                                   size);
}

static void free_rpcmem(void *p)
{
    if (p) rpcmem_free(p);
}

/* ------------------------------------------------------------------ */
/*  Class labels (must match DSP-side / training order)                */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/*  Preprocessing – identical to the Python / x86 C++ version         */
/* ------------------------------------------------------------------ */
static const float MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float STD[3]  = {0.229f, 0.224f, 0.225f};

static void preprocess(const cv::Mat &frame,
                       float         *out_chw)
{
    cv::Mat img;
    cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);

    int h = img.rows;
    int w = img.cols;

    const int resize_size = 256;
    const int crop_size   = RESNET_INPUT_H;

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

    const int spatial = crop_size * crop_size;
    for (int c = 0; c < 3; c++)
        std::memcpy(out_chw + c * spatial,
                    channels[c].ptr<float>(),
                    spatial * sizeof(float));
}

/* ------------------------------------------------------------------ */
/*  Softmax + argmax                                                   */
/* ------------------------------------------------------------------ */
static int argmax_softmax(const float *logits, int n, float *out_score)
{
    float max_val = *std::max_element(logits, logits + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += std::exp(logits[i] - max_val);

    int best = 0;
    float best_p = 0.0f;
    for (int i = 0; i < n; i++) {
        float p = std::exp(logits[i] - max_val) / sum;
        if (p > best_p) {
            best_p = p;
            best   = i;
        }
    }
    if (out_score) *out_score = best_p;
    return best;
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *video_path = "/data/local/tmp/imagenet10_val_video.mp4";
    const char *model_path = "/data/local/tmp/resnet18_imagenet10.bin";

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) video_path = argv[2];

    /* ---- Initialise FastRPC shared-memory allocator ----------------- */
    rpcmem_init();

    /* ---- Allocate ION buffers for zero-copy RPC --------------------- */
    uint8_t *input_buf  = alloc_rpcmem(RESNET_INPUT_SIZE);
    uint8_t *output_buf = alloc_rpcmem(RESNET_OUTPUT_SIZE);

    if (!input_buf || !output_buf) {
        std::cerr << "rpcmem_alloc failed\n";
        return 1;
    }

    /* ---- Init the DSP inference engine ------------------------------ */
    AEEResult rc = resnet_infer_init(model_path,
                                     static_cast<int>(strlen(model_path) + 1));
    if (rc != AEE_SUCCESS) {
        std::cerr << "resnet_infer_init failed: " << rc << "\n";
        free_rpcmem(input_buf);
        free_rpcmem(output_buf);
        rpcmem_deinit();
        return 1;
    }

    /* ---- Query model parameters from DSP ---------------------------- */
    int num_classes = 0;
    resnet_infer_get_param(RESNET_PARAM_NUM_CLASSES, &num_classes);
    std::cout << "Model loaded – num_classes=" << num_classes << "\n";

    /* ---- Open video ------------------------------------------------- */
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << "\n";
        resnet_infer_deinit();
        free_rpcmem(input_buf);
        free_rpcmem(output_buf);
        rpcmem_deinit();
        return 1;
    }

    /* ---- Timing statistics ------------------------------------------ */
    double total_infer_ms   = 0.0;
    double max_infer_ms     = 0.0;
    double min_infer_ms     = 1e9;
    int    total_frames     = 0;

    /* ---- Main loop -------------------------------------------------- */
    while (cap.isOpened()) {
        cv::Mat frame;
        if (!cap.read(frame))
            break;

        /* Preprocess on ARM */
        preprocess(frame, reinterpret_cast<float *>(input_buf));

        /* Run inference on DSP via FastRPC */
        int output_len_actual = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        rc = resnet_infer_process(input_buf,  RESNET_INPUT_SIZE,
                                  output_buf, RESNET_OUTPUT_SIZE,
                                  &output_len_actual);

        auto t1 = std::chrono::high_resolution_clock::now();

        if (rc != AEE_SUCCESS) {
            std::cerr << "process failed: " << rc << "\n";
            break;
        }

        double infer_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        total_infer_ms += infer_ms;
        max_infer_ms    = std::max(max_infer_ms, infer_ms);
        min_infer_ms    = std::min(min_infer_ms, infer_ms);
        total_frames++;

        /* Post-process on ARM */
        const float *logits = reinterpret_cast<const float *>(output_buf);
        float score = 0.0f;
        int   pred  = argmax_softmax(logits, RESNET_NUM_CLASSES, &score);

        const std::string &label = CLASS_NAMES[pred];

        /* Overlay on frame */
        char overlay[128];
        snprintf(overlay, sizeof(overlay),
                 "%s (%.1f%%) | %.1f ms",
                 label.c_str(), score * 100.0f, infer_ms);

        cv::putText(frame, overlay,
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        cv::imshow("Snapdragon DSP Inference", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    /* ---- Print timing summary --------------------------------------- */
    if (total_frames > 0) {
        std::cout << "\n====== DSP Inference Timing ======\n"
                  << "Frames     : " << total_frames                      << "\n"
                  << "Avg (ms)   : " << total_infer_ms / total_frames     << "\n"
                  << "Max (ms)   : " << max_infer_ms                      << "\n"
                  << "Min (ms)   : " << min_infer_ms                      << "\n"
                  << "==================================\n";
    }

    /* ---- Cleanup ---------------------------------------------------- */
    resnet_infer_deinit();
    free_rpcmem(input_buf);
    free_rpcmem(output_buf);
    rpcmem_deinit();

    return 0;
}
