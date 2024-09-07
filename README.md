# smart-surveillance-ml-analysis
ML Analysis pipeline for the Smart Surveillance System

## Dependencies
 * Python 3.9+
 * NumPy
 * Ultralytics
 * OpenCV
 * PyTorch
 * PyTorch Geometric
 * [rtmlib](https://github.com/Tau-J/rtmlib)
 * [Onnxruntime](https://onnxruntime.ai/)

## Environment
 * YOLO_MODEL=yolov10m.pt
 * RTMO_MODEL_URL=[RTMO-M](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip)

## changes from the original idea
 - switch OpenPose to RTMO/YOLO-pose because OpenPose is really old, outdated and unsupported
 - instead of manual feature selection and extraction (which also adds significant time and computational 
 overhead), maybe a ConvLSTM architecture can automatically learn feature representations and save the first and
 second stage of the pipeline (although context clues are still crucially important) [article 1](https://www.mdpi.com/1424-8220/16/1/115),
 [article 2](https://www.mdpi.com/1424-8220/17/11/2556)

## TODO
 - ensure all models are on the GPU if possible
 - write about the tracking algorithms used in YOLO
 - [variable input size for neural network](https://stats.stackexchange.com/a/138760)
 - [Comparative Analysis of OpenPose, PoseNet, and MoveNet](https://iieta.org/journals/ts/paper/10.18280/ts.390111)
 - write about the [RTMO](https://arxiv.org/pdf/2312.07526) model; [additional info](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)
 - write a comparison between [YOLO-pose](https://docs.ultralytics.com/tasks/pose) (two-stage top-down detector) and RTMO (one-stage detector)
 - write about ONNX Runtime
 - write about [MoviNet](https://arxiv.org/pdf/2103.11511) as an alternative to ResNet18_3D
 - write about [LSTM vs Transformers](https://deep-learning-mit.github.io/staging/blog/2023/time-series-lstm-transformer/)
 - write about [Visual Transformers (ViT)](https://arxiv.org/pdf/2102.05095)
 - write about two-stream convolutional networks
 - write about nvidia containers and container runtime plugin, k8s node plugin
 - might write about improvements in [YOLOv9](https://learnopencv.com/yolov9-advancing-the-yolo-legacy/) and YOLOv10

## Miscellaneous 
 - Objects interaction https://arxiv.org/pdf/1612.00222
 - Overkill but still beneficial to write about: [Hybrid LSTM-Transformer architecture](https://www.nature.com/articles/s41598-024-55483-x)
 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8659462/
 - Very relevant to [Abnormal crowd behaviour detection](https://ieeexplore.ieee.org/abstract/document/5206641)
  and [Sensor-based Human Activity Recognition](https://ieeexplore.ieee.org/abstract/document/9333470)
 - The whole system basically represents a (Supervised) [Anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection) system
 - https://www.ibm.com/topics/anomaly-detection
 - probably not useful, but still might be valuable: [xLSTM](https://arxiv.org/html/2406.04303v1)
 - probably not useful, as it is mainly used for predictions: [ConvLSTM](https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7)
 - [useless?](https://arxiv.org/pdf/1711.09577v2)
 - Writing a video to disk: `cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))`

### Potential Benefits of SAGPooling:
- **Focus on Relevant Substructures**: SAGPooling dynamically selects a subset of important nodes based on their
 features and the graph structure. This can potentially allow the network to focus more on regions within the frame
 (graph) that are more relevant for detecting suspicious behavior.
- **Reduction of Complexity**: By reducing the number of nodes, SAGPooling can decrease the computational burden,
 which might be beneficial when processing a large number of frames (graphs), as typically encountered in video data.
- **Enhanced Learning Dynamics**: Through its attention mechanism, SAGPooling may help in learning more
 discriminative features, which could be highly descriptive and beneficial for downstream temporal modeling with LSTMs.

### Implementing SAGPooling While Retaining Fixed-Sized Output:
To utilize SAGPooling while still having a fixed-sized output for LSTM suitability, you can enforce a structured
scenario wherein after SAGPooling - which typically reduces graph size - you apply a global pooling operation that
condenses the information from the pooled graph into a fixed-sized vector. This way, you merge enhanced features from
important nodes into a consistent representation suitable for sequence processing.

### Considerations:
- **Temporal Discontinuity**: Care must be taken to ensure that the SAGPooling operation doesn’t introduce
  discontinuities in the temporal dimension. The selection of nodes should ideally be stable across frames to
  preserve temporal coherence. This is sometimes challenging with dynamic pooling mechanisms, and may require careful
  tuning, or supplementing with position or frame-reference features.
- **Tuning Pooling Ratios**: The pooling ratio and other hyperparameters (like attention specifics in SAGPooling)
  need to be tuned based on the specific characteristics of your data and what constitutes “suspicious behavior”.
