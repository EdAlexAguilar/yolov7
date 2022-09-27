import torch
assert torch.cuda.is_available(), "CUDA not available"
"""
---YOLO imports---
Requires:
1) system $YOLO_PARENT variable with 'yolov7' folder
2) folder should have the weights "yolov7.pt"
"""
#import sys
#import os
#sys.path.append(os.path.expandvars('$YOLO_PARENT'))
import yolov7
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression
from yolov7.utils.torch_utils import select_device, TracedModel

# WEIGHTS = os.path.expandvars('$YOLO_PARENT/yolov7/yolov7.pt')
WEIGHTS = f'{yolov7.__path__[0]}/yolov7.pt'

class YOLODetector:
    def __init__(self, weights=WEIGHTS, img_size=640, trace=True, device='',
                 conf_threshold=0.25, iou_threshold=0.45):
        self.img_size = img_size
        self.device = device
        self.weights = weights
        self.trace = trace
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=stride)  # check img_size
        if self.trace:
            self.model = TracedModel(self.model, self.device, self.img_size) # will JIT the function
        if self.half:
            self.model.half()  # to FP16

        # Run inference once for good luck from ML Gods
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once

    def detect(self, image):
        img = torch.from_numpy(image).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = self.model(img) # todo: check augment
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold) # todo: only apply NMS for certain classes
        return pred

if __name__=='__main__':
    y = YOLOv7()