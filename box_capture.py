from bdb import Breakpoints
import cv2
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt

#img_bgr = cv2.imread('test2.jpg')
 
def capture():
    cap = cv2.VideoCapture(0)
    key = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            #cv2.imshow('original', frame)
            #frame = cv2.rectangle(frame, (125, 125),(510, 360), (0, 255, 255), 3)
            #blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            #orgin = cv2.cvtColor(frame)
            origin = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            retval, bin = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            max_area = 0
            max_index = -1
            
            index = -1
            for i in contours:
                area = cv2.contourArea(i)
                index = index + 1
                if area > max_area:
                    max_area = area
                    max_index = index

            if max_index == -1: # 검출된 contour가 없으면
                break

            # 원본 이미지에 컨투어 표시
            cv2.drawContours(frame, contours, max_index, (0, 255, 0), 3)
            
            # 컨투어를 둘러싸는 가장 작은 사각형 그리기
            cnt = contours[max_index]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box1 = np.int0(box)
            box2 = np.float64(box)
            result = cv2.drawContours(frame, [box1], 0, (0, 0, 255), 2)
            #result2 = cv2.drawContours(frame, [box], 0, (255, 255, 255), 2)
            cv2.circle(result, (100, 100), 10, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(result, (100, 380), 10, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(result, (530, 100), 10, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(result, (530, 380), 10, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.imshow('Camera Window', result)

            if (box1[0][0] >= 100 and box1[2][0] <= 530 and box1[0][1] >= 100 and box1[2][1] <= 380):
                key = key + 1
            else:
                key = 0

            print("key : ", key)

            if(key > 30):
                cv2.imwrite("opencv.jpg", origin)
                img_bgr = cv2.imread('opencv.jpg')

                #pngImageFile = Image.open("test_last.jpg")

                #pos = (box1[0][0] + 15, box1[0][1] + 15)

                #colorTuple = pngImageFile.getpixel(pos)

                #print(colorTuple)
                
                for i in range(480):
                    for j in range(640):
                        if not (box2[0][1] - 15 < i < box2[2][1] + 15 and box2[0][0] - 15 < j < box2[2][0] + 15):
                            img_bgr[i, j, :] = (170, 170, 170)
                cv2.imwrite('pre_address.jpg', img_bgr)

                ImageFile = cv2.imread("pre_address.jpg", cv2.IMREAD_GRAYSCALE)
                thresh_np = np.zeros_like(ImageFile)
                thresh_np[ImageFile > 80] = 255
                #thresh_np[ImageFile <= 50] = 0
                ret,thresh_cv = cv2.threshold(ImageFile, 80, 255,cv2.THRESH_BINARY)

                cv2.imwrite("address.jpg", thresh_cv)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()       
    cv2.destroyAllWindows()