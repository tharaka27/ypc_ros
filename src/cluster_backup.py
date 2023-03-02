#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
import sys
import pcl
import pcl.pcl_visualization
import ros_numpy
import matplotlib.pyplot as plt
import tf
from mpl_toolkits.mplot3d import Axes3D


from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2

# TODO provide CompressedImages capaility for output_image_topic


i = 0
class ClusterDetector():
    def __init__(self): 
        #self._3dray = None
        self.image_data = None
        self.depth_data = None

        #get the masked image
        self.image_sub = rospy.Subscriber(
            rospy.get_param("~output_image_topic"), Image, self.image_callback , queue_size=1
        )

        # get the raw depth image
        self.depth_sub = rospy.Subscriber(
            rospy.get_param("~raw_depth_topic"), Image, self.depth_callback, queue_size=1
        )

        # self.depth_pub = rospy.Publisher(
        #     rospy.get_param("~segmented_depth_topic"), Image, queue_size=1
        # )

        # self.pc_sub = rospy.Subscriber(
        #     rospy.get_param("~input_pc_topic"), PointCloud2, self.pointcloud_callback , queue_size=1
        # )
        # self.pc_sub = rospy.Subscriber(
        #     rospy.get_param("~camera_info_topic"), CameraInfo, self.camera_callback , queue_size=1
        # )
        # self.pc_pub = rospy.Publisher(
        #     rospy.get_param("~segmented_pc_topic"), PointCloud2, queue_size=1
        # )


        # self.camera_model = PinholeCameraModel()
        # self.cloud = pcl.PointCloud
        # self.listner = tf.TransformListener()
        # self.visualize()
        

    # def visualize(self):
    #     rospy.sleep(5)
    #     pc = ros_numpy.numpify(self.data)
    #     filter(lambda v: v==v , pc) #remove all NaN values
    #     height = pc.shape[0]
    #     width = pc.shape[1]
    #     #np_points = np.zeros((height * width, 3), dtype=np.float32)
    #     #np_points[:, 0] = np.resize(pc['x'], height * width)
    #     #np_points[:, 1] = np.resize(pc['y'], height * width)
    #     #np_points[:, 2] = np.resize(pc['z'], height * width)    
        

    #     # cloud = pcl.PointCloud()
    #     # visual = pcl.pcl_visualization.CloudViewing() 
    #     # cloud.from_array(np.array(np_points, dtype=np.float32))
    #     if( self._3dray is not None):
    #         pass

    #     #visual.ShowColorCloud(cloud)
    #     #visual.ShowMonochromeCloud(cloud)
    #     #v = True
    #     #while v:
    #     #    v = not(visual.WasStopped()) 

    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     ax.plot_wireframe(pc['x'], pc['y'], pc['z'], color='black')
    #     ax.plot([0, self._3dray[0] * 1], [0, self._3dray[1] * 1], [0, self._3dray[2] * 1])
    #     ax.plot([0, self._3dray2[0] * 1], [0, self._3dray2[1] * 1], [0, self._3dray2[2] * 1])
    #     ax.plot([0, self._3dray3[0] * 1], [0, self._3dray3[1] * 1], [0, self._3dray3[2] * 1])
    #     ax.plot([0, self._3dray4[0] * 1], [0, self._3dray4[1] * 1], [0, self._3dray4[2] * 1])
    #     ax.plot([0, self._3dray5[0] * 1], [0, self._3dray5[1] * 1], [0, self._3dray5[2] * 1])
    #     print(self.camera_model.project3dToPixel((self._3dray4[0] * 1, self._3dray4[1] * 1, self._3dray4[2] * 1)))
    #     ax.set_title('wireframe')
    #     plt.show()

    def image_callback(self, data: Image):
        #print(data)
        self.image_data = data
        # # rospy.sleep(10)
        # bridge = CvBridge()
        # cv_mat = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        # print(np.amax(cv_mat))
        # print(np.amin(cv_mat))
        # # rec_mat = np.zeros((cv_mat.shape), dtype=cv_mat.dtype)
        # # self.camera_model.rectifyImage(cv_mat, rec_mat)
        # # print(cv_mat.shape)
        # # print(rec_mat.shape)
        # cv2.imshow("test", cv_mat)
        # cv2.waitKey(0)

    def depth_callback(self, data:Image):
        self.depth_data = data
        
        if(not(self.image_data is None)):
            bridge = CvBridge()
            cv_mat = bridge.imgmsg_to_cv2(self.image_data, desired_encoding="bgr8")
            cv_mat = cv2.cvtColor(cv_mat, cv2.COLOR_BGR2GRAY)
            ret,thresh1 = cv2.threshold(cv_mat,30,255,cv2.THRESH_BINARY)
            dep_mat = bridge.imgmsg_to_cv2(self.depth_data)
            cv_image_array = np.array(dep_mat, dtype = np.dtype('f8'))
            #cv_image_array = cv2.resize(cv_image_array, (640,640), interpolation = cv2.INTER_AREA)
            cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
            
            res = cv2.bitwise_and(cv_image_norm,cv_image_norm,mask = thresh1)
            cv2.imshow("depth", res)
            cv2.imshow("imagee" , cv_mat)
                #cv2.imshow("depth", dep_mat)

            

            cv2.waitKey(10)
        

            

    # def pointcloud_callback(self, data:PointCloud2):
    #     self.data = data
        

    # def camera_callback(self, data:CameraInfo):
    #     self.camera_model.fromCameraInfo(data)
        
    #     self._3dray = self.camera_model.projectPixelTo3dRay((0, 0))
    #     self._3dray2 = self.camera_model.projectPixelTo3dRay((0, 180))
    #     self._3dray3 = self.camera_model.projectPixelTo3dRay((640, 0))
    #     self._3dray4 = self.camera_model.projectPixelTo3dRay((640, 180))
    #     self._3dray5 = self.camera_model.projectPixelTo3dRay((320, 90))
        
    #     try:
    #         (trans, rot ) = self.listner.lookupTransform('/base_link', '/camera_color_frame',rospy.Time(0))
    #         print(trans, rot)
    #     except Exception as e:
    #         print(e)
    #     print(self.camera_model.fullProjectionMatrix())
    #     print(self.camera_model.fullIntrinsicMatrix())
    #     print(self.camera_model.distortionCoeffs())
    #     print(self.camera_model.rotationMatrix())

    #     print(self.camera_model.tfFrame())

    #     #self._camera_center = self.camera_model.
    #     #print(self._3dray, )


if __name__ == "__main__":

    rospy.init_node("cluster", anonymous=True)
    detector = ClusterDetector()
    rospy.spin()
