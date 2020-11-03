# -*- coding: utf-8 -*-
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils

#透视矫正
def perspective_transformation(img):
    # 读取图像，做灰度化、高斯模糊、膨胀、Canny边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # edged = cv2.Canny(dilate, 75, 200)
    edged = cv2.Canny(dilate, 73, 200, 3)

    cv2.namedWindow("enhanced", 0);
    cv2.resizeWindow("enhanced", 1280, 960);
    cv2.imshow("enhanced", edged)
    cv2.waitKey(0)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 判断是OpenCV2还是OpenCV3
    docCnt = None

    # 确保至少找到一个轮廓
    if len(cnts) > 0:
        # 按轮廓大小降序排列
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            # 近似轮廓
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 如果我们的近似轮廓有四个点，则确定找到了纸
            if len(approx) == 4:
                docCnt = approx
                break

    # 对原始图像应用四点透视变换，以获得纸张的俯视图
    paper = four_point_transform(img, docCnt.reshape(4, 2))
    return paper


image = cv2.imread('/Users/zhang/test2.JPG')


img = perspective_transformation(image)
cv2.namedWindow("enhanced",0);
cv2.resizeWindow("enhanced", 1280, 960);
cv2.imshow("enhanced",img)
cv2.waitKey(0)



