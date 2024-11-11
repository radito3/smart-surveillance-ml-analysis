# smart-surveillance-ml-analysis
ML Analysis pipeline for the Smart Surveillance System

## Dependencies
 * Python 3.12
 * NumPy
 * Ultralytics
 * OpenCV Headless
 * PyTorch
 * PyTorch Geometric

## changes from the original idea
 - switch OpenPose to RTMO/YOLO-pose because OpenPose is really old, outdated and unsupported
 - instead of manual feature selection and extraction (which also adds significant time and computational 
 overhead), maybe a ConvLSTM architecture can automatically learn feature representations and save the two-stage
  approach of the current pipeline (although context clues are still crucially important) [article 1](https://www.mdpi.com/1424-8220/16/1/115),
 [article 2](https://www.mdpi.com/1424-8220/17/11/2556)

## TODO
 - write about the tracking algorithms used in YOLO
 - [variable input size for neural network](https://stats.stackexchange.com/a/138760)
 - [Comparative Analysis of OpenPose, PoseNet, and MoveNet](https://iieta.org/journals/ts/paper/10.18280/ts.390111)
 - [OpenPose vs MediaPipe](https://maureentkt.medium.com/selecting-your-2d-real-time-pose-estimation-models-7d0777bf935f)
 - write a comparison between pose detectors (YOLO-pose, RTMO, AlphaPose, DEKR, BlazePose, etc.)
 - write about [MoviNet](https://arxiv.org/pdf/2103.11511) and [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) as alternatives to ResNet18_3D
 - write about [LSTM vs Transformers](https://deep-learning-mit.github.io/staging/blog/2023/time-series-lstm-transformer/)
 - write about [Visual Transformers (ViT)](https://arxiv.org/pdf/2102.05095)
 - write about two-stream convolutional networks
 - write about nvidia containers and container runtime plugin, k8s node plugin
 - write about training loss functions, optimizers (SGD, Adam) and metrics (accuracy, precision, recall, F1-score, ROC-AUC)
 - useful info about general principles of [HAR](https://www.sciencedirect.com/science/article/pii/S187705092200045X?ref=cra_js_challenge&fr=RR-1)
 - [Real-world YOLO applications](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai) 

## Miscellaneous 
 - Objects interaction https://arxiv.org/pdf/1612.00222
 - Overkill but still beneficial to write about: [Hybrid LSTM-Transformer architecture](https://www.nature.com/articles/s41598-024-55483-x)
 - [Human Behavior Recognition](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8659462/)
 - Very relevant to [Abnormal crowd behaviour detection](https://ieeexplore.ieee.org/abstract/document/5206641)
  and [Sensor-based Human Activity Recognition](https://ieeexplore.ieee.org/abstract/document/9333470)
 - The whole system basically represents a (Supervised) [Anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection) system
 - https://www.ibm.com/topics/anomaly-detection
 - probably not useful, but still might be valuable: [xLSTM](https://arxiv.org/html/2406.04303v1)
 - probably not useful, as it is mainly used for predictions: [ConvLSTM](https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7)
 - [AlphaPose](https://arxiv.org/pdf/2211.03375)
 - [Spatiotemporal 3D CNNs](https://arxiv.org/pdf/1711.09577v2)
 - Writing a video to disk: `cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))`

### Potential Benefits of SAGPooling:
- **Focus on Relevant Substructures**: SAGPooling dynamically selects a subset of important nodes based on their
 features and the graph structure. This can potentially allow the network to focus more on regions within the frame
 (graph) that are more relevant for detecting suspicious behavior.
- **Reduction of Complexity**: By reducing the number of nodes, SAGPooling can decrease the computational burden,
 which might be beneficial when processing a large number of frames (graphs), as typically encountered in video data.
- **Enhanced Learning Dynamics**: Through its attention mechanism, SAGPooling may help in learning more
 discriminative features, which could be highly descriptive and beneficial for downstream temporal modeling with LSTMs.

### Considerations:
- **Temporal Discontinuity**: Care must be taken to ensure that the SAGPooling operation doesn’t introduce
  discontinuities in the temporal dimension. The selection of nodes should ideally be stable across frames to
  preserve temporal coherence. This is sometimes challenging with dynamic pooling mechanisms, and may require careful
  tuning, or supplementing with position or frame-reference features.
