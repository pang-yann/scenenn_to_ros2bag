#!/usr/bin/env python
import os
import sys
import argparse
import math

import numpy as np
import cv2

from rclpy.time import Time
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, TopicMetadata, StorageOptions, ConverterOptions

from cv_bridge import CvBridge

from geometry_msgs.msg import TransformStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage

_PARENT_FRAME_ID = "/world"
_CHILD_FRAME_ID = "/scenenn_camera_frame"
_WRITE_RGBD = True

# -*- coding: utf-8 -*-
# transformations.py

# Copyright (c) 2006, Christoph Gohlke
# Copyright (c) 2006-2009, The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


_EPS = np.finfo(float).eps * 4.0
_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def vector_norm(data, axis=None, out=None):
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)

def decompose_matrix(matrix):
    M = np.array(matrix, dtype=np.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0, 0, 0, 1
    if not np.linalg.det(P):
        raise ValueError("Matrix is singular")

    scale = np.zeros((3, ), dtype=np.float64)
    shear = [0, 0, 0]
    angles = [0, 0, 0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))
        M[:, 3] = 0, 0, 0, 1
    else:
        perspective = np.array((0, 0, 0, 1), dtype=np.float64)

    translate = M[3, :3].copy()
    M[3, :3] = 0

    row = M[:3, :3].copy()
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        scale *= -1
        row *= -1

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        #angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai = ai / 2.0
    aj = aj / 2.0
    ak = ak / 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion


def rgb_path_from_frame(scenenn_path, scene, frame):
    scene_path = os.path.join(scenenn_path, scene)
    image_path = os.path.join(scene_path, 'image')
    return os.path.join(image_path, 'image{0:05d}.png'.format(frame))

def depth_path_from_frame(scenenn_path, scene, frame):
    scene_path = os.path.join(scenenn_path, scene)
    image_path = os.path.join(scene_path, 'depth')
    return os.path.join(image_path, 'depth{0:05d}.png'.format(frame))

def instance_path_from_frame(scenenn_path, scene, frame):
    scene_path = os.path.join(scenenn_path, scene)
    image_path = os.path.join(scene_path, 'mask')
    return os.path.join(image_path, 'mask_image{0:05d}.png'.format(frame))

def timestamp_path(scenenn_path, scene):
    scene_path = os.path.join(scenenn_path, scene)
    return os.path.join(scene_path, 'timestamp.txt')

def trajectory_path(scenenn_path, scene):
    scene_path = os.path.join(scenenn_path, scene)
    return os.path.join(scene_path, 'trajectory.log')

def intrinsics_path(scenenn_path):
    intrinsics_path = os.path.join(scenenn_path, 'intrinsic')
    return os.path.join(intrinsics_path, 'asus.ini')

def parse_timestamps(scenenn_path, scene):
    timestamps = {}
    try:
        with open(timestamp_path(scenenn_path, scene)) as fileobject:
            for line in fileobject:
                ws = line.split()
                timestamps[int(ws[0])] = int(ws[1])
    except IOError:
        print('SceneNN timestamp data not found at location:{0}'.format(
            timestamp_path(scenenn_path, scene)))
        sys.exit('Please ensure you have downloaded the timestamp data.')

    return timestamps

def parse_trajectory(scenenn_path, scene):
    trajectory = {}
    try:
        with open(trajectory_path(scenenn_path, scene)) as fileobject:
            while 1:
                R = np.diag(np.ones(4))
                frame_line = fileobject.readline()
                if not frame_line:
                    break
                frame = int(frame_line.split()[0])
                R[0, :] = fileobject.readline().split()
                R[1, :] = fileobject.readline().split()
                R[2, :] = fileobject.readline().split()
                R[3, :] = fileobject.readline().split()
                trajectory[frame] = R
    except IOError:
        print('SceneNN trajectory data not found at location:{0}'.format(
            trajectory_path(scenenn_path, scene)))
        sys.exit('Please ensure you have downloaded the trajectory data.')
    return trajectory

def parse_intrinsics(scenenn_path):
    intrinsics = {}
    try:
        with open(intrinsics_path(scenenn_path)) as fileobject:
            for line in fileobject:
                ws = line.split()
                intrinsics[ws[0]] = float(ws[1])
    except IOError:
        print('SceneNN intrinsics data not found at location: {0}'.format(
            intrinsics_path(scenenn_path)))
        sys.exit('Please ensure you have downloaded the intrinsics data.')
    return intrinsics


def camera_intrinsic_transform(intrinsics):
    pixel_width = intrinsics['depth_width']
    pixel_height = intrinsics['depth_height']

    fx = intrinsics['fx']
    fy = intrinsics['fy']

    pixel_width = intrinsics['depth_width']
    pixel_height = intrinsics['depth_height']

    camera_intrinsics = np.zeros((3, 4))
    camera_intrinsics[2, 2] = 1.0
    camera_intrinsics[0, 0] = fx
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = fy
    camera_intrinsics[1, 2] = pixel_height / 2.0
    return camera_intrinsics

def parse_camera_info(scenenn_path):
    intrinsics = parse_intrinsics(scenenn_path)
    camera_intrinsic_matrix = camera_intrinsic_transform(intrinsics)

    camera_info = CameraInfo()
    camera_info.height = int(intrinsics['depth_height'])
    camera_info.width = int(intrinsics['depth_width'])

    camera_info.distortion_model = "plumb_bob"
    camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    camera_info.r = np.ndarray.flatten(np.identity(3))
    camera_info.k = np.ndarray.flatten(camera_intrinsic_matrix[:, :3])
    camera_info.p = np.ndarray.flatten(camera_intrinsic_matrix)

    return camera_info

def parse_tf_message(transform, timestamp):
    scale, shear, angles, transl, persp = decompose_matrix( transform)
    rotation = quaternion_from_euler(*angles)

    ts = TransformStamped()
    ts.header.stamp = timestamp
    ts.header.frame_id = _PARENT_FRAME_ID
    ts.child_frame_id = _CHILD_FRAME_ID
    ts.transform.translation.x = transl[0]
    ts.transform.translation.y = transl[1]
    ts.transform.translation.z = transl[2]
    ts.transform.rotation.x = rotation[0]
    ts.transform.rotation.y = rotation[1]
    ts.transform.rotation.z = rotation[2]
    ts.transform.rotation.w = rotation[3]

    tf_msg = TFMessage()
    tf_msg.transforms.append(ts)
    return tf_msg

def convert(scenenn_path, scene, writer):
    camera_info = parse_camera_info(scenenn_path) # Set camera information and model.
    camera_model = PinholeCameraModel()
    camera_model.fromCameraInfo(camera_info)

    timestamps = parse_timestamps(scenenn_path, scene)
    trajectory = parse_trajectory(scenenn_path, scene)

    header = Header(frame_id=_CHILD_FRAME_ID) # Initialize some vars.
    cvbridge = CvBridge()

    frame = 1
    while (frame < len(timestamps)):
        ros_timestamp = Time(nanoseconds=(timestamps[frame]+1)*1e3) # shift by 1microsecond to avoid 0.00
        header.stamp = ros_timestamp.to_msg()

        try: 
            transform = trajectory[frame - 1]
        except KeyError:
            sys.exit(
                'It is common for transform data to be missing in the last' \
                'few frames of a SceneNN sequence. Stopping reading frames.'
            )
            
        tf_msg = parse_tf_message(transform, header.stamp)
        writer.write('/tf', serialize_message(tf_msg), ros_timestamp.nanoseconds)
        
        # Write the RGBD data.
        if (_WRITE_RGBD):
            rgb_path = rgb_path_from_frame(scenenn_path, scene, frame)
            rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            bgr_msg = cvbridge.cv2_to_imgmsg(rgb_image, "8UC3")
            bgr_msg.encoding = "bgr8" # opencv reads as BGR by default
            bgr_msg.header = header
            writer.write('/camera/rgb/image_raw', serialize_message(bgr_msg), ros_timestamp.nanoseconds)

            depth_path = depth_path_from_frame(scenenn_path, scene, frame)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_msg = cvbridge.cv2_to_imgmsg(depth_image, "16UC1")
            depth_msg.header = header
            writer.write('/camera/depth/image_raw', serialize_message(depth_msg), ros_timestamp.nanoseconds)

            camera_info.header = header
            writer.write('/camera/rgb/camera_info', serialize_message(camera_info), ros_timestamp.nanoseconds)
            writer.write('/camera/depth/camera_info', serialize_message(camera_info), ros_timestamp.nanoseconds)

        sec, nsec = ros_timestamp.seconds_nanoseconds()

        print("Dataset timestamp: " + '{:4}'.format(sec) + "." +
            '{:09}'.format(nsec) + "     Frame: " +
            '{:3}'.format(frame) + " / " + str(len(timestamps)))
        
        frame += 1
        
    print(f"Finished writing Scene {scene}.")

def init_ros2bag_writer(output_bag_path):
    writer = SequentialWriter()
    
    storage_options = StorageOptions(
        uri=output_bag_path,
        storage_id='sqlite3')
    converter_options = ConverterOptions('', '')

    rgb_topic = TopicMetadata(
        name='/camera/rgb/image_raw',
        type='sensor_msgs/msg/Image',
        serialization_format='cdr')
    
    rgb_info_topic = TopicMetadata(
        name='/camera/rgb/camera_info',
        type='sensor_msgs/msg/CameraInfo',
        serialization_format='cdr')
    
    depth_topic = TopicMetadata(
        name='/camera/depth/image_raw',
        type='sensor_msgs/msg/Image',
        serialization_format='cdr')
    
    depth_info_topic = TopicMetadata(
        name='/camera/depth/camera_info',
        type='sensor_msgs/msg/CameraInfo',
        serialization_format='cdr')
    
    tf_topic = TopicMetadata(
        name='/tf',
        type='tf2_msgs/msg/TFMessage',
        serialization_format='cdr')

    writer.open(storage_options, converter_options)
    writer.create_topic(rgb_topic)
    writer.create_topic(rgb_info_topic)
    writer.create_topic(depth_topic)
    writer.create_topic(depth_info_topic)
    writer.create_topic(tf_topic)

    return writer

def main(args=None):
    scenenn_path = "/home/schaeffler/workspace/workspace-fyp/datasets/scenenn_to_ros2bag/scenenn_data"
    scene = "231"
    output_bag_path = "bags"

    if not os.path.isdir(os.path.join(scenenn_path, scene)):
        print("SceneNN scene data not found at {0}".format(
            os.path.join(scenenn_path, scene)))
        sys.exit('Please ensure you have downloaded the scene data.')

    try:
        convert(scenenn_path, scene, output_bag_path)
    except (KeyboardInterrupt):
        print('Conversion interrupted by user.')
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='''%(prog)s [-h] --scenenn-path PATH --scene-id ID --output-bag-dir PATH''',
        description='Write data from a SceneNN scene to a rosbag.')
    parser.add_argument(
        "--scenenn-path",
        required=True,
        help="path to the scenenn_data folder",
        metavar="PATH")
    parser.add_argument(
        "--scene-id",
        required=True,
        help="select the scene with id ID",
        metavar="ID")
    parser.add_argument(
        "--output-bag-dir",
        required=True,
        help="path to the output bag directory",
        metavar="PATH")

    args = parser.parse_args()
    scenenn_path = args.scenenn_path
    scene_id = args.scene_id
    output_bag_dir = args.output_bag_dir
    output_bag_path = os.path.join(output_bag_dir, scene_id)

    if not os.path.isdir(os.path.join(scenenn_path, scene_id)):
        print("SceneNN scene data not found at {0}".format(
            os.path.join(scenenn_path, scene_id)))
        sys.exit('Please ensure you have downloaded the scene data.')
        
    if not os.path.isdir(output_bag_dir):
        print("Creating output bag directory at {0}".format(
            os.path.abspath(output_bag_dir)))
        os.makedirs(output_bag_dir)
        
    try:
        writer = init_ros2bag_writer(output_bag_path)
        convert(scenenn_path, scene_id, writer)
    except Exception as e:
        import traceback
        traceback.print_exc()