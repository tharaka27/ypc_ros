#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
import copy

from sensor_msgs.msg import Image, CompressedImage


# TODO allow compressed image inputs 

class ClusterDetector():
    def __init__(self): 
        self.image_data = None
        self.depth_data = None
        self.bridge = CvBridge()
        self.view_image = rospy.get_param("~view_image")

        #get the masked image
        self.image_sub = rospy.Subscriber(
            rospy.get_param("~output_mask_topic"), Image, self.image_callback , queue_size=1
        )

        # get the raw depth image
        self.depth_sub = rospy.Subscriber(
            rospy.get_param("~input_depth_topic"), Image, self.depth_callback, queue_size=1
        )

        self.depth_pub = rospy.Publisher(
            rospy.get_param("~output_depth_topic"), Image, queue_size=1
        )
        

    def image_callback(self, data: Image):
        base = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.image_data = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    def depth_callback(self, data:Image):
        self.depth_data = data

        send = Image()
        send = copy.deepcopy(self.depth_data)
        
        
        if(not(self.image_data is None)):
            # type convertion from depth image to cv::Mat 
            depth_mat = self.bridge.imgmsg_to_cv2(self.depth_data)
            cv_depth = np.array(depth_mat, dtype = np.dtype('u2')) 
            
            # apply masking to the depth image
            res = cv2.bitwise_and(cv_depth,cv_depth,mask = self.image_data)
            
            # visualize images
            if self.view_image:
                cv2.imshow("depth", res)
                cv2.imshow("color_mask" , res)
                cv2.waitKey(1)

            send.data = self.bridge.cv2_to_imgmsg(res, encoding="16UC1").data

            self.depth_pub.publish(send)
        


if __name__ == "__main__":

    rospy.init_node("cluster", anonymous=True)
    detector = ClusterDetector()
    rospy.spin()
