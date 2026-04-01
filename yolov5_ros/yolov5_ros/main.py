import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

import torch
import torch.backends.cudnn as cudnn

# === YOLOv5 submodule (works with --symlink-install and full install) ===
yolov5_root = str(Path(__file__).resolve().parent.parent / "yolov5")
if yolov5_root not in sys.path:
    sys.path.insert(0, yolov5_root)

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (
    LOGGER,
    check_img_size,
    check_imshow,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox
from std_msgs.msg import Header
from cv_bridge import CvBridge


class yolov5_demo:
    def __init__(
        self,
        weights,
        data,
        imagez_height,
        imagez_width,
        conf_thres,
        iou_thres,
        max_det,
        device,
        view_img,
        classes,
        agnostic_nms,
        line_thickness,
        half,
        dnn,
    ):
        self.weights = weights
        self.data = data
        self.imagez_height = imagez_height
        self.imagez_width = imagez_width
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.line_thickness = line_thickness
        self.half = half
        self.dnn = dnn

        self.s = str()

        self.load_model()

    def load_model(self):
        imgsz = (self.imagez_height, self.imagez_width)

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(
            self.weights,
            device=self.device,
            dnn=self.dnn,
            data=self.data,
            fp16=self.half,
        )
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(imgsz, s=stride)

        self.stride = stride
        self.imgsz = imgsz

        # ─────────────────────────────────────────────────────────────
        # FORCE class names from your Argoverse.yaml (this was the missing piece)
        # ─────────────────────────────────────────────────────────────
        if self.data and Path(self.data).exists():
            try:
                with open(self.data, errors="ignore") as f:
                    data_dict = yaml.safe_load(f)
                yaml_names = data_dict.get("names")
                if yaml_names is not None:
                    # Support both list and dict format in yaml
                    if isinstance(yaml_names, dict):
                        self.names = yaml_names
                    else:
                        self.names = {i: name for i, name in enumerate(yaml_names)}
                    # Only override model if lengths match (recommended)
                    if hasattr(self.model, "names") and len(self.model.names) == len(yaml_names):
                        self.model.names = self.names
                    LOGGER.info(f"✅ Loaded {len(self.names)} class names from {self.data}")
                else:
                    LOGGER.warning("⚠️ No 'names' key found in yaml — using model defaults")
            except Exception as e:
                LOGGER.warning(f"⚠️ Could not load class names from {self.data}: {e}. Using model defaults.")
        else:
            LOGGER.info(f"✅ Using {len(self.names)} class names embedded in the model weights")

        # Warmup
        self.model.warmup(imgsz=(1 if pt or getattr(self.model, "triton", False) else 1, 3, *imgsz))

        self.dt, self.seen = [0.0, 0.0, 0.0], 0

    def image_callback(self, image_raw):
        class_list = []
        confidence_list = []
        x_min_list = []
        y_min_list = []
        x_max_list = []
        y_max_list = []

        # Preprocess exactly like the new dataloaders.py
        img = letterbox(image_raw, self.imgsz, stride=self.stride)[0]

        # HWC → CHW, BGR → RGB
        img = img.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(img)

        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            self.classes,
            self.agnostic_nms,
            max_det=self.max_det,
        )
        self.dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):
            im0 = image_raw.copy()
            self.s += f"{i}: "
            self.s += "%gx%g " % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

            if len(det):
                # Rescale boxes - new function name
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    c_int = int(c)
                    n = (det[:, 5] == c).sum()
                    # SAFE lookup — prevents KeyError
                    class_name = self.names.get(c_int, f"unknown_class_{c_int}")
                    self.s += f"{n} {class_name}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    # SAFE lookup again
                    class_name = self.names.get(c, f"unknown_class_{c}")
                    label = f"{class_name} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    class_list.append(class_name)
                    confidence_list.append(float(conf))
                    x_min_list.append(float(xyxy[0].item()))
                    y_min_list.append(float(xyxy[1].item()))
                    x_max_list.append(float(xyxy[2].item()))
                    y_max_list.append(float(xyxy[3].item()))

            im0 = annotator.result()
            if self.view_img:
                cv2.imshow("yolov5", im0)
                cv2.waitKey(1)

            return class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list

        # no detections
        return class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list


class yolov5_ros(Node):
    def __init__(self):
        super().__init__("yolov5_ros")

        self.bridge = CvBridge()

        self.pub_bbox = self.create_publisher(BoundingBoxes, "yolov5/bounding_boxes", 10)
        self.pub_image = self.create_publisher(Image, "yolov5/image_raw", 10)

        self.sub_image = self.create_subscription(Image, "image_raw", self.image_callback, 10)

        # parameters
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

        self.declare_parameter("weights", str(ROOT) + "/config/yolov5s.pt")
        self.declare_parameter("data", str(Path(yolov5_root)) + "/data/coco128.yaml")
        self.declare_parameter("imagez_height", 640)
        self.declare_parameter("imagez_width", 640)
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("max_det", 1000)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("view_img", True)
        self.declare_parameter("classes", None)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("line_thickness", 2)
        self.declare_parameter("half", False)
        self.declare_parameter("dnn", False)

        self.weights = self.get_parameter("weights").value
        self.data = self.get_parameter("data").value
        self.imagez_height = self.get_parameter("imagez_height").value
        self.imagez_width = self.get_parameter("imagez_width").value
        self.conf_thres = self.get_parameter("conf_thres").value
        self.iou_thres = self.get_parameter("iou_thres").value
        self.max_det = self.get_parameter("max_det").value
        self.device = self.get_parameter("device").value
        self.view_img = self.get_parameter("view_img").value
        self.classes = self.get_parameter("classes").value
        self.agnostic_nms = self.get_parameter("agnostic_nms").value
        self.line_thickness = self.get_parameter("line_thickness").value
        self.half = self.get_parameter("half").value
        self.dnn = self.get_parameter("dnn").value

        self.yolov5 = yolov5_demo(
            self.weights,
            self.data,
            self.imagez_height,
            self.imagez_width,
            self.conf_thres,
            self.iou_thres,
            self.max_det,
            self.device,
            self.view_img,
            self.classes,
            self.agnostic_nms,
            self.line_thickness,
            self.half,
            self.dnn,
        )

    def yolovFive2bboxes_msgs(self, bboxes: list, scores: list, cls: list, img_header: Header):
        bboxes_msg = BoundingBoxes()
        bboxes_msg.header = img_header
        for i in range(len(scores)):
            one_box = BoundingBox()
            one_box.xmin = int(bboxes[0][i])
            one_box.ymin = int(bboxes[1][i])
            one_box.xmax = int(bboxes[2][i])
            one_box.ymax = int(bboxes[3][i])
            one_box.probability = float(scores[i])
            one_box.class_id = cls[i]
            bboxes_msg.bounding_boxes.append(one_box)
        return bboxes_msg

    def image_callback(self, image: Image):
        image_raw = self.bridge.imgmsg_to_cv2(image, "bgr8")
        class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list = (
            self.yolov5.image_callback(image_raw)
        )

        msg = self.yolovFive2bboxes_msgs(
            bboxes=[x_min_list, y_min_list, x_max_list, y_max_list],
            scores=confidence_list,
            cls=class_list,
            img_header=image.header,
        )
        self.pub_bbox.publish(msg)

        self.pub_image.publish(image)


def ros_main(args=None):
    rclpy.init(args=args)
    yolov5_node = yolov5_ros()
    rclpy.spin(yolov5_node)
    yolov5_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    ros_main()