# smart-surveillance-ml-analysis
ML Analysis pipeline for the Smart Surveillance System

## Dependencies
 * Python 3
 * NumPy
 * Scikit-learn
 * Ultralytics
 * OpenCV
 * Keras
 * [rtmlib](https://github.com/Tau-J/rtmlib)
 * [Onnxruntime](https://onnxruntime.ai/)

## Environment
 * YOLO_MODEL=yolov8m.pt
 * RTMO_MODEL_URL=[RTMO-M](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip)
 * KERAS_BACKEND=tensorflow

## changes from the original idea
 - switch OpenPose to RTMO/YOLO-pose because OpenPose is really old, outdated and unsupported
 - instead of manual feature selection and extraction (which also adds significant time and computational 
 overhead), maybe a ConvLSTM architecture can automatically learn feature representations and save the first and
 second stage of the pipeline (although context clues are still crucially important) [article 1](https://www.mdpi.com/1424-8220/16/1/115),
 [article 2](https://www.mdpi.com/1424-8220/17/11/2556)

## TODO
 - keep the ID of the object (person) that is perceived as suspicious
throughout the ML pipeline and add a red bounding box in the dashboard video feed if so
 - [Comparative Analysis of OpenPose, PoseNet, and MoveNet](https://iieta.org/journals/ts/paper/10.18280/ts.390111)
 - write about the [RTMO](https://arxiv.org/html/2312.07526v1) model; [additional info](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)
 - write a comparison between [YOLO-pose](https://docs.ultralytics.com/tasks/pose) (two-stage top-down detector) and RTMO (one-stage detector)
 - write about ONNX Runtime
 - write about [LSTM vs Transformers](https://deep-learning-mit.github.io/staging/blog/2023/time-series-lstm-transformer/)
 - write about [Visual Transformers (ViT)](https://arxiv.org/pdf/2102.05095)
 - write about two-stream convolutional networks
 - write about nvidia containers and container runtime plugin, k8s node plugin
 - might write about improvements in [YOLOv9](https://learnopencv.com/yolov9-advancing-the-yolo-legacy/)
 - https://gist.github.com/dariodip/4e0133eaa8733e4206ccdb48e7af6a90

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
