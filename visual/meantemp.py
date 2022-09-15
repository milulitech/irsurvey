from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance
from scipy.spatial.distance import cosine
import math
image = Image.open("C:/Users/walte/OneDrive/Documents/Research/Anomaly/scale271.jpg")
palette = []
width, height  = image.size
for y in range(width):
	RGB = image.getpixel((y, 10))
	palette.append(RGB)
Tmin = 10
Tmax = 20

T = []
T.append(Tmin)
step = (Tmax - Tmin)/len(palette)
for j in range(len(palette) - 1):
	T.append(T[j] + step)
black_template = cv2.imread('C:/DRLMAV/Project/Src/grpc/Black_template.jpg')
w, h = black_template.shape[::-1]
gray_template = cv2.cvtColor(black_template, cv2.COLOR_BGR2GRAY)
histogram_template = cv2.calcHist([gray_template], [0], None, [256], [0, 256])
thermal = cv2.imread("C:/Users/walte/OneDrive/Documents/Research/Anomaly/Detections16062022/img_new3.jpg")
thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
img  = Image.open("C:/Users/walte/OneDrive/Documents/Research/Anomaly/Detections16062022/img_new3.jpg")
height, width, channels = thermal.shape
TA = []
for y in range(0, int(height - h), h):
        for x in range(0, int(width - w), w):
            x = int(x)
            y = int(y)
            window = thermal[y:y+h, x:x+w]
            histogram_window = cv2.calcHist([window], [0], None, [256], [0, 256])
            dist = cosine(histogram_window, histogram_template)
            if dist >= 0.8:
                break
            else:
		Tff = []
		if y+h <= height and x+w <= width:
			for p in range(y, y+h):
				for q in range(x, x+w):
					distances = []
                			dist = []
                			for l in range(0, len(palette)):
                				RGB = img.getpixel((q, p))
                            			dist = [distance.euclidean(RGB, palette[l])]
                            			distances.append(dist)
                            		index = np.argmin(distances)
                            		#dmin = distances[index]
                            		#color = palette[index]
                            		Tpart = round(T[index], 2)
                            		Tff.append(T[index])
		Ta = np.mean(Tff)
		TA.append(Ta)

Tmean = np.mean(TA)
