/*
 * resnet_infer_imp.cpp
 *
 * DSP-side implementation of the resnet_infer FastRPC interface.
 * This code runs on the Hexagon DSP and uses QNN (Qualcomm AI Engine
 * Direct) to execute the ResNet18 model.
 *
 * Build output: libresnet_infer_skel.so  (pushed to /vendor/lib/rfsa/dsp/
 *               or /usr/lib/rfsa/adsp/ on the device)
 *
 * Prerequisites:
 *   - Convert the ONNX model to QNN format:
 *       qnn-onnx-converter --input_network resnet18_imagenet10.onnx \
 *                          --output_path   resnet18_imagenet10.cpp
 *       qnn-model-lib-generator -c resnet18_imagenet10.cpp \
 *                               -b resnet18_imagenet10.bin \
 *                               -t hexagon-v68
 *   - The resulting context binary (.bin) or model .so is what init() loads.
 */

#include "resnet_infer.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

/* ------------------------------------------------------------------ */
/*  QNN headers                                                        */
/* ------------------------------------------------------------------ */
#include "QnnInterface.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnBackend.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"

/* ------------------------------------------------------------------ */
/*  Internal state                                                     */
/* ------------------------------------------------------------------ */
namespace {

struct InferCtx {
    bool initialised = false;

    /* QNN handles */
    Qnn_BackendHandle_t  backend  = nullptr;
    Qnn_DeviceHandle_t   device   = nullptr;
    Qnn_ContextHandle_t  context  = nullptr;
    Qnn_GraphHandle_t    graph    = nullptr;
    Qnn_ProfileHandle_t  profile  = nullptr;

    /* QNN interface function table – populated from the backend library */
    QnnInterface_t *qnn_interface = nullptr;

    /* Runtime parameters */
    int confidence_threshold = 500;   /* default 0.5 in fixed-point /1000 */
};

InferCtx g_ctx;

/* ------------------------------------------------------------------ */
/*  QNN helper: load context binary produced by qnn-context-binary-    */
/*  generator and retrieve the single graph inside it.                 */
/* ------------------------------------------------------------------ */
AEEResult load_context_binary(const char *path)
{
    /* Read the .bin file into memory */
    FILE *fp = fopen(path, "rb");
    if (!fp)
        return AEE_EFAILED;

    fseek(fp, 0, SEEK_END);
    size_t size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t *buf = (uint8_t *)malloc(size);
    if (!buf) {
        fclose(fp);
        return AEE_ENOMEMORY;
    }

    if (fread(buf, 1, size, fp) != size) {
        free(buf);
        fclose(fp);
        return AEE_EFAILED;
    }
    fclose(fp);

    /*
     * Create a QNN context from the serialised binary.
     * The exact API calls depend on your QNN SDK version; adapt as needed.
     */
    auto &iface = *g_ctx.qnn_interface;

    Qnn_ContextBinaryConfig_t bin_cfg;
    memset(&bin_cfg, 0, sizeof(bin_cfg));

    if (iface.contextCreateFromBinary(
            g_ctx.backend,
            g_ctx.device,
            nullptr,                  /* engine config */
            buf,
            size,
            &g_ctx.context,
            g_ctx.profile) != QNN_SUCCESS) {
        free(buf);
        return AEE_EFAILED;
    }
    free(buf);

    /* Retrieve the first (and only) graph from the context */
    uint32_t num_graphs = 0;
    const QnnGraph_Info_t *graph_info = nullptr;

    if (iface.contextGetGraphInfo(
            g_ctx.context, &graph_info, &num_graphs) != QNN_SUCCESS ||
        num_graphs == 0) {
        return AEE_EFAILED;
    }

    if (iface.graphRetrieve(
            g_ctx.context,
            graph_info[0].graphName,
            &g_ctx.graph) != QNN_SUCCESS) {
        return AEE_EFAILED;
    }

    return AEE_SUCCESS;
}

} /* anonymous namespace */

/* ================================================================== */
/*  PUBLIC INTERFACE IMPLEMENTATION                                    */
/* ================================================================== */

AEEResult resnet_infer_init(const char *model_path,
                            int         model_path_len)
{
    (void)model_path_len;

    if (g_ctx.initialised)
        return AEE_SUCCESS;

    /*
     * 1. Initialise the QNN HTP (Hexagon Tensor Processor) backend.
     *
     *    In a real deployment you would dlopen() the backend .so that ships
     *    with the QNN SDK  (libQnnHtp.so / libQnnHtpStub.so) and resolve
     *    QnnInterface_getProviders() to fill g_ctx.qnn_interface.
     *
     *    Skeleton example (adapt paths / error handling):
     *
     *        void *lib = dlopen("libQnnHtp.so", RTLD_NOW);
     *        auto getProviders = (Qnn_InterfaceGetProvidersFn_t)
     *                             dlsym(lib, "QnnInterface_getProviders");
     *        const QnnInterface_t **providers = nullptr;
     *        uint32_t n = 0;
     *        getProviders(&providers, &n);
     *        g_ctx.qnn_interface = (QnnInterface_t *)providers[0];
     */

    /* ---- placeholder: initialise QNN backend & device --------------- */
    /*
    auto &iface = *g_ctx.qnn_interface;
    iface.backendCreate(nullptr, nullptr, &g_ctx.backend);
    iface.deviceCreate(nullptr, &g_ctx.device);
    iface.profileCreate(g_ctx.backend, QNN_PROFILE_LEVEL_BASIC, &g_ctx.profile);
    */

    /* 2. Load the context binary / model */
    AEEResult rc = load_context_binary(model_path);
    if (rc != AEE_SUCCESS)
        return rc;

    g_ctx.initialised = true;
    return AEE_SUCCESS;
}

AEEResult resnet_infer_deinit(void)
{
    if (!g_ctx.initialised)
        return AEE_SUCCESS;

    auto &iface = *g_ctx.qnn_interface;

    if (g_ctx.context)
        iface.contextFree(g_ctx.context, g_ctx.profile);
    if (g_ctx.device)
        iface.deviceFree(g_ctx.device);
    if (g_ctx.backend)
        iface.backendFree(g_ctx.backend);
    if (g_ctx.profile)
        iface.profileFree(g_ctx.profile);

    memset(&g_ctx, 0, sizeof(g_ctx));
    return AEE_SUCCESS;
}

AEEResult resnet_infer_set_param(int param_id, int value)
{
    switch (param_id) {
    case RESNET_PARAM_CONFIDENCE_THR:
        g_ctx.confidence_threshold = value;
        return AEE_SUCCESS;
    default:
        return AEE_EBADPARM;
    }
}

AEEResult resnet_infer_get_param(int param_id, int *value)
{
    if (!value)
        return AEE_EBADPARM;

    switch (param_id) {
    case RESNET_PARAM_INPUT_WIDTH:     *value = RESNET_INPUT_W;                break;
    case RESNET_PARAM_INPUT_HEIGHT:    *value = RESNET_INPUT_H;                break;
    case RESNET_PARAM_INPUT_CHANNELS:  *value = RESNET_INPUT_C;                break;
    case RESNET_PARAM_NUM_CLASSES:     *value = RESNET_NUM_CLASSES;            break;
    case RESNET_PARAM_CONFIDENCE_THR:  *value = g_ctx.confidence_threshold;    break;
    default:
        return AEE_EBADPARM;
    }
    return AEE_SUCCESS;
}

AEEResult resnet_infer_process(const uint8_t *input_data,
                               int            input_data_len,
                               uint8_t       *output_data,
                               int            output_data_len,
                               int           *output_data_lenout)
{
    if (!g_ctx.initialised)
        return AEE_ENOTINITIALIZED;
    if (!input_data || !output_data || !output_data_lenout)
        return AEE_EBADPARM;
    if (input_data_len < RESNET_INPUT_SIZE)
        return AEE_EBADPARM;
    if (output_data_len < RESNET_OUTPUT_SIZE)
        return AEE_EBADPARM;

    /* ----------------------------------------------------------------
     * Build QNN input tensor
     *
     * The input_data buffer contains the preprocessed CHW float32 tensor
     * produced by the ARM wrapper (resize -> center-crop -> normalise).
     * ---------------------------------------------------------------- */

    Qnn_Tensor_t input_tensor;
    memset(&input_tensor, 0, sizeof(input_tensor));

    uint32_t input_dims[] = {1, RESNET_INPUT_C, RESNET_INPUT_H, RESNET_INPUT_W};
    input_tensor.version           = QNN_TENSOR_VERSION_1;
    input_tensor.v1.dataType       = QNN_DATATYPE_FLOAT_32;
    input_tensor.v1.rank           = 4;
    input_tensor.v1.dimensions     = input_dims;
    input_tensor.v1.clientBuf.data = (void *)input_data;
    input_tensor.v1.clientBuf.dataSize = (uint32_t)input_data_len;

    /* ----------------------------------------------------------------
     * Build QNN output tensor
     * ---------------------------------------------------------------- */

    Qnn_Tensor_t output_tensor;
    memset(&output_tensor, 0, sizeof(output_tensor));

    uint32_t output_dims[] = {1, RESNET_NUM_CLASSES};
    output_tensor.version           = QNN_TENSOR_VERSION_1;
    output_tensor.v1.dataType       = QNN_DATATYPE_FLOAT_32;
    output_tensor.v1.rank           = 2;
    output_tensor.v1.dimensions     = output_dims;
    output_tensor.v1.clientBuf.data = (void *)output_data;
    output_tensor.v1.clientBuf.dataSize = RESNET_OUTPUT_SIZE;

    /* ----------------------------------------------------------------
     * Execute the graph
     * ---------------------------------------------------------------- */

    auto &iface = *g_ctx.qnn_interface;

    Qnn_Tensor_t inputs[]  = {input_tensor};
    Qnn_Tensor_t outputs[] = {output_tensor};

    if (iface.graphExecute(g_ctx.graph,
                           inputs,  1,
                           outputs, 1,
                           g_ctx.profile,
                           nullptr) != QNN_SUCCESS) {
        return AEE_EFAILED;
    }

    *output_data_lenout = RESNET_OUTPUT_SIZE;
    return AEE_SUCCESS;
}
