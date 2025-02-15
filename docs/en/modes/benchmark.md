---
comments: true
description: Learn how to evaluate your YOLOv8 model's performance in real-world scenarios using benchmark mode. Optimize speed, accuracy, and resource allocation across export formats.
keywords: model benchmarking, YOLOv8, Ultralytics, performance evaluation, export formats, ONNX, TensorRT, OpenVINO, CoreML, TensorFlow, optimization, mAP50-95, inference time
---

# Model Benchmarking with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Once your model is trained and validated, the next logical step is to evaluate its performance in various real-world scenarios. Benchmark mode in Ultralytics YOLOv8 serves this purpose by providing a robust framework for assessing the speed and accuracy of your model across a range of export formats.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?start=105"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics Modes Tutorial: Benchmark
</p>

## Why Is Benchmarking Crucial?

- **Informed Decisions:** Gain insights into the trade-offs between speed and accuracy.
- **Resource Allocation:** Understand how different export formats perform on different hardware.
- **Optimization:** Learn which export format offers the best performance for your specific use case.
- **Cost Efficiency:** Make more efficient use of hardware resources based on benchmark results.

### Key Metrics in Benchmark Mode

- **mAP50-95:** For object detection, segmentation, and pose estimation.
- **accuracy_top5:** For image classification.
- **Inference Time:** Time taken for each image in milliseconds.

### Supported Export Formats

- **ONNX:** For optimal CPU performance
- **TensorRT:** For maximal GPU efficiency
- **OpenVINO:** For Intel hardware optimization
- **CoreML, TensorFlow SavedModel, and More:** For diverse deployment needs.

!!! Tip "Tip"

    * Export to ONNX or OpenVINO for up to 3x CPU speedup.
    * Export to TensorRT for up to 5x GPU speedup.

## Usage Examples

Run YOLOv8n benchmarks on all supported export formats including ONNX, TensorRT etc. See Arguments section below for a full list of export arguments.

!!! Example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark on GPU
        benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## Arguments

Arguments such as `model`, `data`, `imgsz`, `half`, `device`, and `verbose` provide users with the flexibility to fine-tune the benchmarks to their specific needs and compare the performance of different export formats with ease.

| Key       | Default Value | Description                                                                                                                                       |
| --------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`   | `None`        | Specifies the path to the model file. Accepts both `.pt` and `.yaml` formats, e.g., `"yolov8n.pt"` for pre-trained models or configuration files. |
| `data`    | `None`        | Path to a YAML file defining the dataset for benchmarking, typically including paths and settings for validation data. Example: `"coco8.yaml"`.   |
| `imgsz`   | `640`         | The input image size for the model. Can be a single integer for square images or a tuple `(width, height)` for non-square, e.g., `(640, 480)`.    |
| `half`    | `False`       | Enables FP16 (half-precision) inference, reducing memory usage and possibly increasing speed on compatible hardware. Use `half=True` to enable.   |
| `int8`    | `False`       | Activates INT8 quantization for further optimized performance on supported devices, especially useful for edge devices. Set `int8=True` to use.   |
| `device`  | `None`        | Defines the computation device(s) for benchmarking, such as `"cpu"`, `"cuda:0"`, or a list of devices like `"cuda:0,1"` for multi-GPU setups.     |
| `verbose` | `False`       | Controls the level of detail in logging output. A boolean value; set `verbose=True` for detailed logs or a float for thresholding errors.         |

## Export Formats

Benchmarks will attempt to run automatically on all possible export formats below.

| Format                                            | `format` Argument | Model                     | Metadata | Arguments                                                            |
| ------------------------------------------------- | ----------------- | ------------------------- | -------- | -------------------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                   | -                 | `yolov8n.pt`              | ✅       | -                                                                    |
| [TorchScript](../integrations/torchscript.md)     | `torchscript`     | `yolov8n.torchscript`     | ✅       | `imgsz`, `optimize`, `batch`                                         |
| [ONNX](../integrations/onnx.md)                   | `onnx`            | `yolov8n.onnx`            | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `batch`             |
| [OpenVINO](../integrations/openvino.md)           | `openvino`        | `yolov8n_openvino_model/` | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TensorRT](../integrations/tensorrt.md)           | `engine`          | `yolov8n.engine`          | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `batch` |
| [CoreML](../integrations/coreml.md)               | `coreml`          | `yolov8n.mlpackage`       | ✅       | `imgsz`, `half`, `int8`, `nms`, `batch`                              |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model`     | `yolov8n_saved_model/`    | ✅       | `imgsz`, `keras`, `int8`, `batch`                                    |
| [TF GraphDef](../integrations/tf-graphdef.md)     | `pb`              | `yolov8n.pb`              | ❌       | `imgsz`, `batch`                                                     |
| [TF Lite](../integrations/tflite.md)              | `tflite`          | `yolov8n.tflite`          | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TF Edge TPU](../integrations/edge-tpu.md)        | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅       | `imgsz`                                                              |
| [TF.js](../integrations/tfjs.md)                  | `tfjs`            | `yolov8n_web_model/`      | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [PaddlePaddle](../integrations/paddlepaddle.md)   | `paddle`          | `yolov8n_paddle_model/`   | ✅       | `imgsz`, `batch`                                                     |
| [NCNN](../integrations/ncnn.md)                   | `ncnn`            | `yolov8n_ncnn_model/`     | ✅       | `imgsz`, `half`, `batch`                                             |

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I benchmark my YOLOv8 model's performance using Ultralytics?

Ultralytics YOLOv8 offers a Benchmark mode to assess your model's performance across different export formats. This mode provides insights into key metrics such as mean Average Precision (mAP50-95), accuracy, and inference time in milliseconds. To run benchmarks, you can use either Python or CLI commands. For example, to benchmark on a GPU:

!!! Example

    === "Python"

    ```python
    from ultralytics.utils.benchmarks import benchmark

    # Benchmark on GPU
    benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
    ```

    === "CLI"

    ```bash
    yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
    ```

For more details on benchmark arguments, visit the [Arguments](#arguments) section.

### What are the benefits of exporting YOLOv8 models to different formats?

Exporting YOLOv8 models to different formats such as ONNX, TensorRT, and OpenVINO allows you to optimize performance based on your deployment environment. For instance:

- **ONNX:** Provides up to 3x CPU speedup.
- **TensorRT:** Offers up to 5x GPU speedup.
- **OpenVINO:** Specifically optimized for Intel hardware.
    These formats enhance both the speed and accuracy of your models, making them more efficient for various real-world applications. Visit the [Export](../modes/export.md) page for complete details.

### Why is benchmarking crucial in evaluating YOLOv8 models?

Benchmarking your YOLOv8 models is essential for several reasons:

- **Informed Decisions:** Understand the trade-offs between speed and accuracy.
- **Resource Allocation:** Gauge the performance across different hardware options.
- **Optimization:** Determine which export format offers the best performance for specific use cases.
- **Cost Efficiency:** Optimize hardware usage based on benchmark results.
    Key metrics such as mAP50-95, Top-5 accuracy, and inference time help in making these evaluations. Refer to the [Key Metrics](#key-metrics-in-benchmark-mode) section for more information.

### Which export formats are supported by YOLOv8, and what are their advantages?

YOLOv8 supports a variety of export formats, each tailored for specific hardware and use cases:

- **ONNX:** Best for CPU performance.
- **TensorRT:** Ideal for GPU efficiency.
- **OpenVINO:** Optimized for Intel hardware.
- **CoreML & TensorFlow:** Useful for iOS and general ML applications.
    For a complete list of supported formats and their respective advantages, check out the [Supported Export Formats](#supported-export-formats) section.

### What arguments can I use to fine-tune my YOLOv8 benchmarks?

When running benchmarks, several arguments can be customized to suit specific needs:

- **model:** Path to the model file (e.g., "yolov8n.pt").
- **data:** Path to a YAML file defining the dataset (e.g., "coco8.yaml").
- **imgsz:** The input image size, either as a single integer or a tuple.
- **half:** Enable FP16 inference for better performance.
- **int8:** Activate INT8 quantization for edge devices.
- **device:** Specify the computation device (e.g., "cpu", "cuda:0").
- **verbose:** Control the level of logging detail.
    For a full list of arguments, refer to the [Arguments](#arguments) section.
