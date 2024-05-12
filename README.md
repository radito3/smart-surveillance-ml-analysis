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

## TODO
 - keep the ID of the object (person) that is perceived as suspicious
throughout the ML pipeline and add a red bounding box in the dashboard video feed if so
 - [Comparative Analysis of OpenPose, PoseNet, and MoveNet](https://iieta.org/journals/ts/paper/10.18280/ts.390111)
 - write about the [RTMO](https://arxiv.org/html/2312.07526v1) model; [additional info](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)
 - write a comparison between [YOLO-pose](https://docs.ultralytics.com/tasks/pose) (two-stage top-down detector) and RTMO (one-stage detector)
 - write about ONNX Runtime
