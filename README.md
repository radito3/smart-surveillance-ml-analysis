# smart-surveillance-ml-analysis
ML Analysis pipeline for the Smart Surveillance System

## Dependencies
 * Python 3
 * NumPy
 * Scikit-learn
 * Ultralytics
 * OpenCV
 * rtmlib

## changes from original idea (name pending)
 - switch OpenPose to RTMO/YOLO-pose because OpenPose is really old, outdated and unsupported

## TODO
 - keep the ID of the object (person) that is perceived as suspicious
throughout the ML pipeline and add a red bounding box in the dashboard video feed if so
 - [Comparative Analysis of OpenPose, PoseNet, and MoveNet](https://iieta.org/journals/ts/paper/10.18280/ts.390111)
