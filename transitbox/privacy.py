"""Face-sized redaction with YuNet and short motion-aware tracking."""
from pathlib import Path
import ctypes
import hashlib
import importlib.util
import math
import urllib.request

import cv2
import numpy as np

from transitbox.privacy_constants import MODEL_COMMIT, MODEL_NAME, MODEL_SHA256, MODEL_URL, VERSION, FACE_PADDING, HOLD_SECONDS, MIN_SCORE, LARGE_BOX_SCORE


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
    def __init__(self, model, device='cpu', threshold=MIN_SCORE, max_side=1280):
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

    def detect(self, image, max_scale=4., scene_guard=True):
        return self.detect_batch([image],max_scale,scene_guard)[0]

    def detect_batch(self, images, max_scale=4., scene_guard=True):
        if not images:
            return []
        if any(image.shape != images[0].shape for image in images):
            return [self.detect(image,max_scale,scene_guard) for image in images]
        batch = len(images)
        image = images[0]
        height, width = image.shape[:2]
        scale = min(max_scale, self.max_side/max(height, width))
        w, h = max(32, round(width*scale)), max(32, round(height*scale))
        pad_w, pad_h = math.ceil(w/32)*32, math.ceil(h/32)*32
        blob = np.zeros((batch,3,pad_h,pad_w), dtype=np.float32)
        for i, image in enumerate(images):
            blob[i,:,:h,:w] = cv2.resize(image, (w,h)).transpose(2,0,1)
        outputs = self.session.run(None, {self.input_name:blob})
        return [self._decode([output.reshape(batch,-1,output.shape[-1])[i] for output in outputs],
                            width,height,w,h,pad_w,scene_guard) for i in range(batch)]

    def _decode(self, outputs, width, height, w, h, pad_w, scene_guard=True):
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
            for box,confidence in zip(decoded.tolist(),score[indices].tolist()):
                if valid_face_box(box,confidence,width,height,scene_guard):
                    boxes.append(box);scores.append(confidence)
        if not boxes:
            return []
        selected = np.asarray(cv2.dnn.NMSBoxes(boxes, scores, self.threshold, .3)).reshape(-1)
        return [tuple(boxes[i]) for i in selected]


def valid_face_box(box, score, width, height, scene_guard=True):
    _x,_y,w,h = box
    if not all(math.isfinite(value) for value in box) or w<3 or h<3 or not .30<=w/h<=1.8:
        return False
    # Low-confidence arm/device detections used to produce oversized boxes.
    large = w*h>width*height*.04 or h>height*.25
    return score >= (LARGE_BOX_SCORE if scene_guard and large else MIN_SCORE)


def expanded_box(box, width, height, padding=FACE_PADDING):
    x,y,w,h = box
    return (max(0, math.floor(x-w*padding)), max(0, math.floor(y-h*padding)),
            min(width, math.ceil(x+w*(1+padding))), min(height, math.ceil(y+h*(1+padding))))


class TemporalFaces:
    """Predict briefly missing face positions; never replace faces with body boxes."""
    def __init__(self, hold_seconds=HOLD_SECONDS):
        self.hold_seconds = hold_seconds
        self.tracks = []

    def update(self, timestamp, boxes):
        old = [track for track in self.tracks if timestamp-track['seen']<=self.hold_seconds]
        matched = set()
        current = []
        for box in boxes:
            x,y,w,h = box
            candidates = []
            for i,track in enumerate(old):
                if i in matched:
                    continue
                px,py,pw,ph = self.predict(track,timestamp)
                distance = math.hypot(x+w/2-px-pw/2,y+h/2-py-ph/2)
                if .5<=w/pw<=2 and .5<=h/ph<=2 and distance<=max(w,h,pw,ph):
                    candidates.append((distance,i))
            velocity=(0.,0.)
            if candidates:
                i=min(candidates)[1];matched.add(i)
                previous=old[i];px,py,pw,ph=previous['box'];dt=timestamp-previous['seen']
                if dt>0:
                    limit=max(w,h)*4
                    velocity=(float(np.clip((x-px)/dt,-limit,limit)),float(np.clip((y-py)/dt,-limit,limit)))
            current.append({'seen':timestamp,'box':box,'velocity':velocity})
        current.extend(item for i,item in enumerate(old) if i not in matched)
        self.tracks = current
        return [self.predict(track,timestamp) for track in current]

    @staticmethod
    def predict(track,timestamp):
        x,y,w,h=track['box'];vx,vy=track['velocity'];dt=max(0,timestamp-track['seen'])
        return (x+vx*dt,y+vy*dt,w,h)

def redact(image, face_boxes):
    """Destroy facial detail inside the full mask; leave surrounding pixels intact."""
    height,width = image.shape[:2]
    mask = np.zeros((height,width), np.uint8)
    regions = [expanded_box(box,width,height) for box in face_boxes]
    result = image.copy()
    for x1,y1,x2,y2 in regions:
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(width,x2),min(height,y2)
        if x2>x1 and y2>y1:
            mask[y1:y2,x1:x2] = 255
            w,h=x2-x1,y2-y1
            roi=image[y1:y2,x1:x2]
            tiny=cv2.resize(roi,(min(4,w),min(4,h)),interpolation=cv2.INTER_AREA)
            tiny=cv2.GaussianBlur(tiny,(3,3),sigmaX=1.)
            result[y1:y2,x1:x2]=cv2.resize(tiny,(w,h),interpolation=cv2.INTER_LINEAR)
    return result, int(np.count_nonzero(mask))
