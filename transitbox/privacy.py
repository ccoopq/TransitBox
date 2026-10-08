"""Face redaction using YuNet, enlarged masks and conservative head fallbacks."""
from pathlib import Path
import ctypes
import hashlib
import importlib.util
import math
import urllib.request

import cv2
import numpy as np

from transitbox.privacy_constants import MODEL_COMMIT, MODEL_NAME, MODEL_SHA256, MODEL_URL, VERSION


def get_model(path):
    path = Path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != MODEL_SHA256:
            raise ValueError('Face detector download checksum mismatch')
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError('Face detector checksum mismatch')
    return path


def preload_cuda():
    spec = importlib.util.find_spec('nvidia')
    if not spec or not spec.submodule_search_locations:
        return
    for folder, filename in (
        ('cuda_runtime', 'libcudart.so.12'), ('cublas', 'libcublasLt.so.12'),
        ('cublas', 'libcublas.so.12'), ('cufft', 'libcufft.so.11'),
        ('curand', 'libcurand.so.10'), ('cuda_nvrtc', 'libnvrtc.so.12'),
        ('cudnn', 'libcudnn_ops.so.9'), ('cudnn', 'libcudnn_adv.so.9'),
        ('cudnn', 'libcudnn_cnn.so.9'), ('cudnn', 'libcudnn_graph.so.9'),
        ('cudnn', 'libcudnn_engines_runtime_compiled.so.9'),
        ('cudnn', 'libcudnn_engines_precompiled.so.9'),
        ('cudnn', 'libcudnn_heuristic.so.9'), ('cudnn', 'libcudnn.so.9')):
        for root in spec.submodule_search_locations:
            lib = Path(root)/folder/'lib'/filename
            if lib.is_file():
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                break


class FaceDetector:
    """YuNet outputs decoded using the equations in OpenCV FaceDetectorYN."""
    def __init__(self, model, device='cpu', threshold=.28, max_side=1280):
        import onnxruntime as ort
        self.threshold = threshold
        self.max_side = max_side
        if device == 'cuda':
            preload_cuda()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search':'HEURISTIC',
                      'gpu_mem_limit':2*1024*1024*1024, 'use_tf32':False}), 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        import onnx
        graph = onnx.load(str(get_model(model)))
        # Only make the input batch dimension symbolic; weights/operators stay intact.
        # YuNet's output reshape flattens batches into anchors, decoded separately below.
        graph.graph.input[0].type.tensor_type.shape.dim[0].ClearField('dim_value')
        graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch'
        self.session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=providers)
        if device == 'cuda' and 'CUDAExecutionProvider' not in self.session.get_providers():
            raise RuntimeError('CUDA face detection requested but unavailable')
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image):
        return self.detect_batch([image])[0]

    def detect_batch(self, images):
        if not images:
            return []
        if any(image.shape != images[0].shape for image in images):
            return [self.detect(image) for image in images]
        batch = len(images)
        image = images[0]
        height, width = image.shape[:2]
        scale = min(4., self.max_side/max(height, width))
        w, h = max(32, round(width*scale)), max(32, round(height*scale))
        pad_w, pad_h = math.ceil(w/32)*32, math.ceil(h/32)*32
        blob = np.zeros((batch,3,pad_h,pad_w), dtype=np.float32)
        for i, image in enumerate(images):
            blob[i,:,:h,:w] = cv2.resize(image, (w,h)).transpose(2,0,1)
        outputs = self.session.run(None, {self.input_name:blob})
        return [self._decode([output.reshape(batch,-1,output.shape[-1])[i] for output in outputs],
                            width,height,w,h,pad_w) for i in range(batch)]

    def _decode(self, outputs, width, height, w, h, pad_w):
        boxes, scores = [], []
        for i, stride in enumerate((8,16,32)):
            score = np.sqrt(np.clip(outputs[i].reshape(-1),0,1)*np.clip(outputs[i+3].reshape(-1),0,1))
            indices = np.flatnonzero(score >= self.threshold)
            if not len(indices):
                continue
            offsets = outputs[i+6].reshape(-1,4)[indices]
            cols = pad_w//stride
            centers = (np.column_stack((indices%cols, indices//cols))+offsets[:,:2])*stride
            sizes = np.exp(np.clip(offsets[:,2:], -10,10))*stride
            xy = centers-sizes/2
            decoded = np.column_stack((xy[:,0]*width/w, xy[:,1]*height/h,
                                       sizes[:,0]*width/w, sizes[:,1]*height/h))
            boxes.extend(decoded.tolist())
            scores.extend(score[indices].tolist())
        if not boxes:
            return []
        selected = np.asarray(cv2.dnn.NMSBoxes(boxes, scores, self.threshold, .3)).reshape(-1)
        return [tuple(boxes[i]) for i in selected]


def expanded_box(box, width, height, padding=.40):
    x,y,w,h = box
    return (max(0, math.floor(x-w*padding)), max(0, math.floor(y-h*padding)),
            min(width, math.ceil(x+w*(1+padding))), min(height, math.ceil(y+h*(1+padding))))


def head_box(bounds, width, height):
    x1,y1,x2,y2 = bounds
    w,h = (x2-x1)*width,(y2-y1)*height
    # Wider than the head and extending into the upper torso for occluded/tiny faces.
    return (max(0, math.floor(x1*width-w*.12)), max(0, math.floor(y1*height-h*.04)),
            min(width, math.ceil(x2*width+w*.12)), min(height, math.ceil(y1*height+h*.42)))


class TemporalFaces:
    """Hold missed detections briefly without leaving trails behind detected faces."""
    def __init__(self, hold_seconds=1.):
        self.hold_seconds = hold_seconds
        self.tracks = []

    def update(self, timestamp, boxes):
        old = [(seen,box) for seen,box in self.tracks if timestamp-seen<=self.hold_seconds]
        matched = set()
        current = []
        for box in boxes:
            x,y,w,h = box
            candidates = []
            for i,(_seen,previous) in enumerate(old):
                if i in matched:
                    continue
                px,py,pw,ph = previous
                distance = math.hypot(x+w/2-px-pw/2,y+h/2-py-ph/2)
                if distance<=max(w,h,pw,ph):
                    candidates.append((distance,i))
            if candidates:
                matched.add(min(candidates)[1])
            current.append((timestamp,box))
        current.extend(item for i,item in enumerate(old) if i not in matched)
        self.tracks = current
        return [box for _,box in current]


def uncovered_heads(face_boxes, head_boxes):
    result = []
    for x1,y1,x2,y2 in head_boxes:
        width,height = x2-x1,y2-y1
        covered = any(x>=x1 and y>=y1 and x+w<=x2 and y+h<=y2 and
                      abs(x+w/2-(x1+x2)/2)<=width*.25 and
                      y+h/2-y1<=height*.65 for x,y,w,h in face_boxes)
        if not covered:
            result.append((x1,y1,x2,y2))
    return result


def redact(image, face_boxes, head_boxes=()):
    """Destroy facial detail inside the full mask; leave surrounding pixels intact."""
    height,width = image.shape[:2]
    mask = np.zeros((height,width), np.uint8)
    regions = [expanded_box(box,width,height,padding=.15 if box[2]*box[3]>width*height*.04 else .40)
               for box in face_boxes]+list(head_boxes)
    for x1,y1,x2,y2 in regions:
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(width,x2),min(height,y2)
        if x2>x1 and y2>y1:
            mask[y1:y2,x1:x2] = 255
    result = image.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1,count):
        x,y,w,h,area = stats[i]
        roi = image[y:y+h,x:x+w]
        # Four color cells per dimension, followed by strong smoothing.
        tiny = cv2.resize(roi, (min(4,w),min(4,h)), interpolation=cv2.INTER_AREA)
        tiny = cv2.GaussianBlur(tiny, (3,3), sigmaX=1.)
        blurred = cv2.resize(tiny, (w,h), interpolation=cv2.INTER_LINEAR)
        selected = labels[y:y+h,x:x+w] == i
        result[y:y+h,x:x+w][selected] = blurred[selected]
    return result, int(np.count_nonzero(mask))
