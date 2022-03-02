# -*- coding: utf-8 -*-
from skimage.metrics import structural_similarity
import cv2   #4.5.2
import numpy as np
import time


#一、加载视频流
#加载视频流    0为本机摄像头  1为视频采集卡
cap0 = cv2.VideoCapture()
cap0 = cv2.VideoCapture(1+ cv2.CAP_DSHOW)  # 视频流
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 分辨率
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

# 1280分辨率   数字区域偏移量参数
traffic_x=750
traffic_y=205
traffic_w=475
traffic_h=280

#二、模板项处理
# 记录起始帧
i=0
while(i<=50):
    if(cap0.isOpened()):
        ret, frameA = cap0.read()
        if ret==True:
            #记录模板项
            cv2.imwrite('start.jpg', frameA)
        pass
    pass
    i+=1
pass
#数字区域灰度处理
grayA = cv2.cvtColor(frameA[traffic_y:traffic_y+traffic_h,traffic_x:traffic_x+traffic_w], cv2.COLOR_BGR2GRAY)
#白底图片反向最优阈值二值化 -->  黑底白字便于识别
ret1, binary_img1 = cv2.threshold(grayA, 0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#闭运算
kernel1 = np.ones((5, 5), np.uint8)
close_img1 = cv2.morphologyEx(binary_img1 , cv2.MORPH_CLOSE, kernel1)
#膨胀运算
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilation_img1 = cv2.dilate(close_img1, element1, iterations=2)
#边缘检测
cnts1, hierarchy1 = cv2.findContours(dilation_img1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#测试经过膨胀运算的图片
# cv2.imshow("a",dilation_img1)
#测试模板项文字识别区域
# n=0
# for c in cnts1:
#    (x, y, w, h) = cv2.boundingRect(c)
#    cv2.rectangle(frameA, (x+traffic_x,y+traffic_y),(x+w+traffic_x,y+h+traffic_y), (0,255, 0), 2)
#    cv2.imshow("b", frameA[traffic_y:traffic_y + traffic_h, traffic_x:traffic_x + traffic_w])
#    n=n+1
# pass
#测试输出检测到的文本个数
# print("模板项的文本区域有"+str(n)+"个")

#三、每一帧处理
while(cap0.isOpened()):
    ret,frameB=cap0.read()
    if ret==True:
        kuang = []
        #记录每一帧的起始时间
        start = time.time()
        #数字区域
        imageB=frameB
        #数字区域灰度处理
        grayB = cv2.cvtColor(frameB[traffic_y:traffic_y+traffic_h,traffic_x:traffic_x+traffic_w], cv2.COLOR_BGR2GRAY)
        #差分处理  -->模板项与处理帧比对   diff为[0, 1]的浮点数 1为相同点
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        #数组转换为范围[0, 256]的8位无符号整数    越不同的地方越黑（0）
        diff = (diff * 255).astype("uint8")
        #白底图片反向最优阈值二值化 -->  黑底白字便于识别
        ret, binary_img = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        #闭运算
        kernel = np.ones((5, 5), np.uint8)
        close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        #膨胀运算
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilation_img = cv2.dilate(close_img, element, iterations=2)
        #边缘检测
        cnts, hierarchy = cv2.findContours(dilation_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (diff_x, diff_y, diff_w, diff_h) = cv2.boundingRect(c)
            for d in cnts1:
                (x, y, w, h) = cv2.boundingRect(d)
                # 判断差异的四点坐标  是否在模板项的文字区域
                if ((x <= diff_x <= x + w and y <= diff_y <= y + h) or (
                        x <= diff_x + diff_w <= x + w and y <= diff_y <= y + h) or (
                        x <= diff_x <= x + w and y <= diff_y + diff_h <= y + h) or (
                        x <= diff_x + diff_w <= x + w and y <= diff_y + diff_h <= y + h)):
                    #如果差分轮廓包含在模板项轮廓中，在当前项中框选所对应的模板项区域
                    cv2.rectangle(frameB, (x + traffic_x, y + traffic_y), (x + w + traffic_x, y + h + traffic_y),(0, 0, 255), 2)
                    #放入数字识别代码
                    # kuang.append((x+traffic_x, y+traffic_y, w, h))
                    # x2 = list(set(kuang))
                    # print(x2)
                    # print("--------")

                pass
            pass
        pass
        # x2 = list(set(kuang))
        # print(x2)
        # print("--------")
        #记录结束时间
        end = time.time()
        #转换为帧率
        frames_num = round(1 / (end - start))
        #将帧率记录在左上角
        cv2.putText(frameB, "FPS {0}".format(frames_num), (10, 30), 1, 1.5, (255, 0, 255), 2)
        #测试整个页面-->二选一
        # cv2.imshow("PC Monitor ALL", frameB)
        #测试文本区域-->二选一
        resize_frameB=cv2.resize(frameB[traffic_y:traffic_y+traffic_h,traffic_x:traffic_x+traffic_w],None,fx=1,fy=1.2,interpolation=cv2.INTER_CUBIC)
        cv2.putText(resize_frameB, "FPS {0}".format(frames_num), (10, 30), 1, 1.5, (255, 0, 255), 2)
        cv2.imshow("PC Monitor TEXT",resize_frameB)
    pass
    #设置速度
    if cv2.waitKey(50)&0xFF==ord("q"):
        break

    pass
pass





cap0.release()
cv2.destroyAllWindows()