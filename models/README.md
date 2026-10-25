# Detector provenance

`efficientdet-lite0.tflite` is the unmodified TensorFlow EfficientDet-Lite0
uint8 COCO detector, metadata version 1. `labelmap.txt` is extracted from its
embedded ZIP metadata. Model and labels are licensed Apache 2.0; the application
remains under the repository's existing MIT license.

- Official model card: https://www.kaggle.com/models/tensorflow/efficientdet/TfLite/lite0-detection-metadata/1
- Official model download: https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite0_detection_metadata_1.tflite
- TensorFlow example download manifest: https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/download_models.gradle
- Kaggle model-instance API returned `licenseName: Apache 2.0`, version 1 on 2026-10-24.
- SHA-256: `2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b`
- Input: uint8 RGB, 1 × 320 × 320 × 3. Outputs: 25 normalized boxes, classes, confidence scores, count.

MobileNetV2 SSD FPN-Lite was the initial candidate. This implementation uses
TensorFlow's directly published mobile detector with an explicit LiteRT tensor
contract, avoiding an unverified SavedModel-to-TFLite conversion. This is not a
claim that EfficientDet outperforms MobileNetV2 on X8; no comparative benchmark
has been run. See the hardware verification record for measured performance.

COCO recognition is limited to its object classes. A bottle label establishes
neither glass/plastic composition nor recyclability. No automatic disposal-bin
mapping is implemented. Trash-specific supervised training requires a separate
labeled dataset, evaluation, license review, and model adapter.
