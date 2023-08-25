# -*- coding:utf-8 -*-
# !/usr/bin/env python
from concurrent.futures import thread
from filecmp import cmp
from glob import glob
from cv2 import THRESH_BINARY
import numpy as np
import cv2
import rospy, cv2, cv_bridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy, LaserScan
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String as StringMsg
from simple_follower.msg import position as PositionMsg
import math

# 全局变量的定义
img_count = 0

# 全局变量控制图像获取和雷达避障的发布优先级
lidar_first = 0  # 0 表示雷达未检测到障碍物； 1 表示检测到障碍物，停止，图像话题不控制小车移动

# 用于判断调整角度
left_k = 0   
right_k = 0

#巡线图像返回值
no_line = 0  #没检测到车道
turn_left = 1  #没检测到左车道，左转
turn_right = 2  #没检测到右车道，右转
str_left = 3  #直行，向左微调
str_right = 4  #直行，向右微调
str = 5  #直行
zhijiao_left = 6  #检测到直角弯，向左转
zhijiao_right = 7  #检测到直角弯，向右转
zebra_have = 9  # 有斑马线
zebra_no =10  #无斑马线
red_light = 11  #红灯
green_light = 12  #绿灯
no_light = 13  



#图像预处理
def process(frame):
    frame=cv2.convertScaleAbs(frame,alpha=1.0,beta=-10)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.Canny(gray, 150, 300)
    h, w = gray.shape
    binary[0:np.int(h / 2 + 50), 0:w] = 0
    # 轮廓查找
    i, contours, hierarchy= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 创建输出用的空白图像
    out_image = np.zeros((h, w), frame.dtype)
    out_image1 = np.zeros((h, w), frame.dtype)
    # 遍历每一个轮廓，进行轮廓分析
    for cnt in range(len(contours)):
        cv2.drawContours(out_image1, contours, cnt, (255), 2, 8)
        p = cv2.arcLength(contours[cnt], True) # 计算轮廓周长
        area = cv2.contourArea(contours[cnt]) # 计算轮廓面积
        x, y, rw, rh = cv2.boundingRect(contours[cnt]) # 获取轮廓的中心坐标以及长、宽
        if p < 5 or area < 10:
            continue
        if y > (h - 50):
            continue
        (x, y), (a, b), angle = cv2.minAreaRect(contours[cnt]) # 计算最小外接矩形角度
        angle = abs(angle)
        if angle < 5 or angle > 160 or angle == 90.0:
            continue
        if len(contours[cnt]) > 5: # contour的长度大于5
            (x, y), (a, b), degree = cv2.fitEllipse(contours[cnt])  # 椭圆拟合
            if degree>160 or 80<degree<100:
                continue
        # 不被以上的条件剔除的，在创建的空白图像上绘制该轮廓
        cv2.drawContours(out_image, contours, cnt, (255), 2, 8)
    return out_image


#巡线车道线检测
def line_detection(image, masking):
    lines = cv2.HoughLinesP(masking, 1, np.pi / 180, 70, minLineLength=20, maxLineGap=10)
    left_lines = []
    right_lines = []
    zhijiao_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0] # 斜率
                y_intercept = parameters[1] # 截距
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if slope < -0.2:
                    left_lines.append((slope, y_intercept))
                elif slope > 0.2:
                    right_lines.append((slope, y_intercept))
                else:
                    zhijiao_lines.append((slope, y_intercept))
    else:
        print('未检测到车道')
        return no_line

    left_k = 0
    right_k = 0
    have_left = 1
    have_right = 1
    if left_lines:
        left_avg = np.average(left_lines, axis=0)
        left_k = left_avg[0]
        left_b = left_avg[1]
    else:
        have_left = 0 #没检测到左
    if right_lines:
        right_avg = np.average(right_lines, axis=0)
        right_k = right_avg[0]
        right_b = right_avg[1]
    else:
        have_right = 0 #没检测到右
    if zhijiao_lines:
        zhijiao_avg = np.average(zhijiao_lines, axis=0)
        zhijiao_k = zhijiao_avg[0]

    if have_left == 1 and have_right == 1:
        # jiehe = left_k + right_k
        # if jiehe > 0.2:
        #     print("向左微调")
        #     return str_left
        # elif jiehe < -0.2:
        #     print("向右微调")
        #     return str_right
        # else:
        #     print("直行")
        #     return str

        # 计算两线交点
        img_count = 0
        x = (right_b*1.0 - left_b*1.0)/(left_k*1.0 - right_k*1.0)
        x0 = image.shape[1] / 2
        cx = x - x0
        cv2.line(image, (int(x), 480), (int(x), 0), (0, 255, 0), 1)
        cv2.line(image, (320, 480), (320, 0), (0, 0, 255), 1)
        if cx < -20:
            print('向左微调')
            return str_left
        elif cx > 20:
            print('向右微调')
            return str_right
        else:
            print('直行')
            return str
    else:
        if have_left == 0 and have_right == 1:
            print('左转：未检测到左车道')
            return turn_left
        elif have_left == 1 and have_right == 0:
            print('右转：未检测到右车道')
            return turn_right
        else:
            # 判断是否有水平线（确定直角弯）
            if len(zhijiao_lines)!= 0:
                if(zhijiao_k > 0):
                    print('左转：检测到直角弯')
                    return zhijiao_left
                else:
                    print('右转：检测到直角弯')
                    return zhijiao_right
                    print('左右为空：未检测到车道')
            return no_line


# 斑马线检测
def Slow(img):
    kernel_Ero = np.ones((3,1),np.uint8)
    kernel_Dia = np.ones((5,1),np.uint8)
    copy_img = img.copy()
    copy_img = cv2.resize(copy_img,(1600,800))
    count=0
    # 图像灰度化
    gray=cv2.cvtColor(copy_img,cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    imgblur=cv2.GaussianBlur(gray,(5,5),10)
    #阈值处理
    mask = np.zeros_like(gray)
    mask[400:800,100:1600]=gray[400:800,100:1600]
    ret,thresh=cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
    #腐蚀
    img_Ero=cv2.erode(thresh,kernel_Ero,iterations=3)
    #膨胀
    img_Dia=cv2.dilate(img_Ero,kernel_Dia,iterations=1)
    #轮廓检测
    i, contouts,h = cv2.findContours(img_Dia,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = contouts
    for i in cnt:
        #坐标赋值
        x,y,w,h = cv2.boundingRect(i)
        #print(x,y,w,h)
        if  w>30 and h>30:
            out=cv2.drawContours(copy_img,i,-1,(0,255,0),3)
            count=count+1
    if count>=5 :
        print("斑马线")
        return zebra_have
    return zebra_no


# 交通灯检测
def is_light(image):
    light = -1
    image = cv2.resize(image, (640, 480))
    img = np.zeros_like(image)
    img[170:470,100:540] = image[170:470,100:540]
    cv2.imshow('img', img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，分离.
    redlow = np.array([170, 100, 150])
    redhigh = np.array([180, 255, 255])
    greenlow = np.array([55, 100, 65])
    greenhigh = np.array([67, 255, 255]) 
    dst1 = cv2.inRange(src=hsv, lowerb=redlow, upperb=redhigh) # HSV高低阈值，提取图像部分区域
    dst2 = cv2.inRange(src=hsv, lowerb=greenlow, upperb=greenhigh) # HSV高低阈值，提取图像部分区域
    cv2.imshow('red', dst1)
    cv2.imshow('green', dst2)
    # 轮廓检测
    contours1, hierarchy= cv2.findContours(dst1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy= cv2.findContours(dst2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 椭圆拟合
    for cnt1 in range(len(contours1)):
        (x, y), (a, b), degree = cv2.fitEllipse(contours1[cnt1])
        if x>0 and y>0:
            light = 1 #红灯
        cv2.drawContours(image, contours1, cnt1, (0,0,255), 3)
    for cnt2 in range(len(contours2)):
        (x, y), (a, b), degree = cv2.fitEllipse(contours2[cnt1])
        if x>0 and y>0:
            light = 2 #绿灯
        cv2.drawContours(image, contours2, cnt2, (0,255,0), 3)
    cv2.imshow('a', image)
    if light == 1:
        print('检测到红灯')
        return red_light
    elif light == 2:
        print('检测到绿灯')
        return green_light
    else:
        print('no')
        return no_light


#节点定义
class Follower:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()

        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw',Image, self.image_callback)
        self.scanSubscriber = rospy.Subscriber('/scan', LaserScan, self.registerScan)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel',Twist, queue_size=1)
    

    def moveCar(self,line,ang):
        twist = Twist()
        twist.linear.x = line
        twist.angular.z = ang
        self.cmd_vel_pub.publish(twist)


    # 雷达避障
    def registerScan(self, scan_data):
        global lidar_first
        flag = 0
        ranges = np.array(scan_data.ranges)
        for i in range(360):
            if i >= 178 and i <= 182:
                if math.isinf(ranges[i]) == True:
                    continue
                if np.any(ranges[i] <= 0.7):
                    flag = 1
                    break
                else:
                    flag = 0
        
        if flag == 1:
            lidar_first = 1
            print('障碍物')
        else:
            lidar_first = 0


    # 图像回调
    def image_callback(self, msg):
        global lidar_first
        global left_k, right_k
        global img_count

        if lidar_first == 1:
            self.moveCar(0.0, 0.0)
        else:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            masking = process(image)
            go_where = line_detection(image, masking)
            slow = Slow(image)
            
            light = is_light(image)

            # 红灯停
            if light == red_light:
                self.moveCar(0.0, 0.0)
            # 绿灯行
            else:
                # 没有检测到斑马线
                if slow == zebra_no:
                    # if go_where == turn_left:  # 根据车道斜率判断转弯
                    #     if abs(right_k) < 0.4:  # 需要拐弯
                    #         self.moveCar(0.15, 0.3)
                    #     else:  # 直行情况下过偏，缓慢调整
                    #         self.moveCar(0.15, 0.15)
                    # elif go_where == turn_right:
                    #     if abs(left_k) < 0.4:
                    #         self.moveCar(0.15, -0.3)
                    #     else:
                    #         self.moveCar(0.15, -0.15)
                    if go_where == turn_left:  # 根据车道斜率判断转弯
                        self.moveCar(0.25, 0.3)
                    elif go_where == turn_right:
                        self.moveCar(0.25, -0.3)
                    elif go_where == str_left:
                        self.moveCar(0.25, 0.13)
                    elif go_where == str_right:
                        self.moveCar(0.25, -0.13)
                    elif go_where == zhijiao_left:
                        self.moveCar(0.15, 0.6)
                    elif go_where == zhijiao_right:
                        self.moveCar(0.15, -0.6)
                    else:
                        self.moveCar(0.3, 0)
                # 检测到斑马线
                else:
                    # if go_where == turn_left:  # 根据车道斜率判断转弯
                    #     if abs(right_k) < 0.4:  # 需要拐弯
                    #         self.moveCar(0.1, 0.3)
                    #     else:  # 直行情况下过偏，缓慢调整
                    #         self.moveCar(0.1, 0.15)
                    # elif go_where == turn_right:
                    #     if abs(left_k) < 0.4:
                    #         self.moveCar(0.1, -0.3)
                    #     else:
                    #         self.moveCar(0.1, -0.15)
                    if go_where == turn_left:  
                        self.moveCar(0.1, 0.3)
                    elif go_where == turn_right:
                        self.moveCar(0.1, -0.3)
                    elif go_where == str_left:
                        self.moveCar(0.1, 0.13)
                    elif go_where == str_right:
                        self.moveCar(0.1, -0.13)
                    else:
                        self.moveCar(0.1, 0)


rospy.init_node('Car')
follower = Follower()
rospy.spin()