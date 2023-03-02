# Point cloud clustering using YOLOv5 ROS
This is a ROS node for using YOLOv5 for real time point cloud segmentation. It uses segmentation provided in the [official YOLOv5 repository](https://github.com/ultralytics/yolov5). This project is adapted from [Base ROS YOLOv5 implementation](https://github.com/mats-robotics/ypc_ros)

![Alt Text](https://github.com/tharaka27/ypc_ros/blob/main/misc/video.gif)

## Installation

### Dependencies
This package is built and tested on Ubuntu 20.04 LTS and ROS Noetic with Python 3.8.

* Clone the packages to ROS workspace and install requirement for YOLOv5 submodule:
```bash
cd <ros_workspace>/src
git clone https://github.com/mats-robotics/detection_msgs.git
git clone --recurse-submodules https://github.com/tharaka27/ypc_ros.git 
cd ypc_ros/src/yolov5
pip3 install -r requirements.txt # install the requirements for yolov5
```
* Build the ROS package:
```bash
cd <ros_workspace>
catkin build ypc_ros # build the ROS package
```
* Make the Python script executable 
```bash
cd <ros_workspace>/src/ypc_ros/src
chmod +x detect.py
chmod +x cluster.py
chmod +x camera_info.py
```

![Alt Text](https://github.com/tharaka27/ypc_ros/blob/main/misc/archi.png)

## Basic usage

* Launch the node:
```bash
roslaunch ypc_ros ypc.launch
```

## Using custom weights and dataset (Working)
* Put your weights into `ypc_ros/src/yolov5`
* Put the yaml file for your dataset classes into `ypc_ros/src/yolov5/data`
* Change related ROS parameters in yolov5.launch: `weights`,  `data`

## Reference
* YOLOv5 official repository: https://github.com/ultralytics/yolov5
* YOLOv3 ROS PyTorch: https://github.com/eriklindernoren/PyTorch-YOLOv3
* Darknet ROS: https://github.com/leggedrobotics/darknet_ros
* YOLOv5 ROS : https://github.com/mats-robotics/yolov5_ros
