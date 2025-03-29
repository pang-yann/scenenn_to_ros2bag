# Interfacing SceneNN and ROS2 (Fork)
Dataset tools for working with the [SceneNN](http://scenenn.net/) dataset and converting its raw sequence data to a ROS2 bag.

This repository was forked from [ethz-asl/scenenn_to_rosbag](https://github.com/ethz-asl/scenenn_to_rosbag) and modified to work with ROS2 as part of my Final Year Project. Since the masked instance images are no longer available via the original scenenn Google Drive, it only converts the RGB-D images and trajectories into the ROS2 bag.

## How to use these tools
1. Follow the instructions in the [scenenn repository](https://github.com/scenenn/scenenn) and download the SceneNN data.
Your `scenenn_data` folder structure should be at the end as follows:

    ```
    scenenn_data
    ├── 123
    │   ├── depth
    │   │   ├── depth00001.png
    │   │   ├── ...
    │   ├── image
    │   │   ├── image00001.png
    │   │   ├── ...
    │   ├── timestamp.txt
    │   ├── trajectory.log
    │   ├── ...
    └── intrinsic
        ├──  asus.ini
        ├──  kinect2.ini
    ```

2. Clone this repository.

    ```bash
    git clone git@github.com:pang-yann/scenenn_to_ros2bag.git
    ```

3. Download dependencies
    ```bash
    sudo apt install ros-humble-cv-bridge
    ```

4. Run the Python script directly to convert data from a SceneNN scene to a ros2bag. The ros2bag will contain a sequence of RGB and depth images and relative transforms.

    ```bash
    python scripts/scenenn_to_ros2bag.py --scenenn-path PATH/TO/scenenn_data --scene-id ID --output-bag-dir PATH/TO/output_bags
    ```

    For example:
    ```bash
    python scripts/scenenn_to_ros2bag.py --scenenn-path ../../../scenenn/download/scenenn_data/ --scene-id 066 --output-bag-dir ./bag/
    ```
    The output bag contains the following topics:
    ```bash
    # RGB and depth images
    /camera/rgb/camera_info         : sensor_msgs/CameraInfo
    /camera/rgb/image_raw           : sensor_msgs/Image
    /camera/depth/camera_info       : sensor_msgs/CameraInfo
    /camera/depth/image_raw         : sensor_msgs/Image        

    # Transform from /scenenn_camera_frame to /world
    /tf                             : tf/tfMessage
    ```
