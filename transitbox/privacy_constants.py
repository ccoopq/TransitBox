"""Public redaction protocol and verified detector source; no runtime dependencies."""
MODEL_COMMIT = '47534e27c9851bb1128ccc0102f1145e27f23f98'
MODEL_NAME = 'face_detection_yunet_2026may.onnx'
MODEL_SHA256 = 'ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0'
MODEL_URL = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_COMMIT}/models/face_detection_yunet/{MODEL_NAME}'
VERSION = 'yunet-face-only-temporal-v2'
FACE_PADDING = .15
HOLD_SECONDS = .5
MIN_SCORE = .35
LARGE_BOX_SCORE = .70
