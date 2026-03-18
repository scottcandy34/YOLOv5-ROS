# YOLOv5-ROS

[YOLOv5](https://github.com/ultralytics/yolov5) + ROS2 object detection package

This program changes the input of detect.py (ultralytics/yolov5) to `sensor_msgs/Image` of ROS2.

<br>

Maybe [this one](https://github.com/Alpaca-zip/ultralytics_ros) is easier to use.

<br>

## Installation

```bash
mkdir -p yolov5_ws/src
cd yolov5_ws/src

git clone https://github.com/scottcandy34/YOLOv5-ROS.git
git clone https://github.com/scottcandy34/bbox_ex_msgs.git

pip3 install -r ./YOLOv5-ROS/requirements.txt --break-system-packages

cd ..
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install

echo "source ~/yolov5_ws/install/local_setup.bash" >> ~/.bashrc
source ~/.bashrc
```

<br>

## Run with QT viewer
```bash
ros2 run yolov5_ros yolov5_ros --ros-args -p view_img:=True
```

## Demo

```bash
cd yolov5_ws/
source ./install/setup.bash
ros2 launch yolov5_ros yolov5s_simple.launch.py
```

<br>


## Requirements
- ROS2 Jazzy (Foxy, Humble, Iron)
- OpenCV 4
- PyTorch
- bbox_ex_msgs

## Topic

### Subscribe
- image_raw (`sensor_msgs/Image`)

### Publish
- yolov5/image_raw : Resized image (`sensor_msgs/Image`)
- yololv5/bounding_boxes : Output BoundingBoxes like darknet_ros_msgs (`bboxes_ex_msgs/BoundingBoxes`)

※ If you want to use `darknet_ros_msgs` , replace `bboxes_ex_msgs` with `darknet_ros_msgs`.

## About YOLOv5 and contributers

- [YOLOv5 : GitHub](https://github.com/ultralytics/yolov5)
- [Glenn Jocher : GitHub](https://github.com/glenn-jocher)

### What is YOLOv5 🚀

YOLOv5 is the most useful object detection program in terms of speed of CPU inference and compatibility with PyTorch.

> Shortly after the release of YOLOv4 Glenn Jocher introduced YOLOv5 using the Pytorch framework.
The open source code is available on GitHub


## About writer
- Ar-Ray : Japanese student.
- Blog (Japanese) : https://ar-ray.hatenablog.com/
- Twitter : https://twitter.com/Ray255Ar
