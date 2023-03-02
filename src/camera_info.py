#!/usr/bin/env python3
import rospy

from sensor_msgs.msg import CameraInfo
import numpy as np
import copy
import math

# TODO provide CompressedImages capaility for output_image_topic
# TODO complete scaling operator properly


class ClusterDetector():
    def __init__(self): 
        
        #get the camera image
        self.image_sub = rospy.Subscriber(
            rospy.get_param("~camera_info_topic"), CameraInfo, self.camera_callback , queue_size=1
        )

        #get the camera image
        self.depth_pub = rospy.Publisher(
            rospy.get_param("~output_camera_info"), CameraInfo, queue_size=1
        )
        
        self.height = 480 #rospy.get_param("~inference_size_h", 480)
        self.width  = 640 #rospy.get_param("~inference_size_w", 640)

    def camera_callback(self, data:CameraInfo):
        #c_info = CameraInfo()
        #c_info = data
        #c_info.height = self.height
        #c_info.width = self.width

        # current_height  = data.height #480
        # current_width = data.width #848
        # new_height = self.height #480
        # new_width = self.width #640

        # print(current_height, new_height, "height")
        # print(current_width, new_width, "width") 
        # height_cn = float(current_height)/float(new_height)
        # width_cn = float(current_width)/float(new_width)

        # print(current_height, new_height, current_width, new_width)

        # ## Need to change camera info K and P to match new size of the image
        # # https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
        # current_k = data.K
        # current_P = data.P

        # k_mat = np.reshape(np.array(list(current_k)), (3,3))
        # a = np.array([[width_cn, 0, 0], [0, height_cn, 0], [0, 0, 1]])

        # new = a * k_mat
        # print(k_mat)
        # print(a)
        # print(new)


        c_info = copy.deepcopy(data)

        self.depth_pub.publish(c_info)


if __name__ == "__main__":

    rospy.init_node("cluster", anonymous=True)
    detector = ClusterDetector()
    rospy.spin()
