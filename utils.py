import cv2
import numpy as np
import os
from constants import *

def save_results(file_name,img):
	image_path = os.path.join(results_folder,file_name)
	cv2.imwrite(image_path,img)

def read_image(file_name):
	image_path = os.path.join(images_folder,file_name)
	img = cv2.imread(image_path)
	return img

def draw_circles(img,circles):
	circles = np.uint16(np.around(circles.astype(np.float32)))
	for i in circles[0,:]:
	    # draw the outer circle
	    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
	    # draw the center of the circle
	    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

def find_hough_circles(img,min_radius,max_radius):
	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=min_radius,maxRadius=max_radius)
	#circles = np.uint16(np.around(circles))
	#print "Number of circles found:"+str(len(circles[0]))
	return circles

def improve_contrast(img):
	clahe = cv2.createCLAHE(clipLimit=200.0, tileGridSize=(8,8))
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	l2 = clahe.apply(l)
	lab = cv2.merge((l2,a,b))
	return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def get_edges(img,min_thresh,max_thresh):
    return cv2.Canny(img,min_thresh,max_thresh)

def dilate(img,kernel,num_iterations):
    return cv2.dilate(img, kernel, iterations=num_iterations)

def threshold(img):
	ret,thresh1 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
	return thresh1

def erode(img):
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(img,kernel,iterations = 1)
	return erosion

def opening(img):
	kernel = np.ones((5,5),np.uint8)
	return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(img):
	kernel = np.ones((5,5),np.uint8)
	return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def sharpen(img):
	kernel = np.zeros((9,9),np.float32)
	kernel[4,4] = 2.0
	boxfilter = np.ones((9,9),np.float32)/81
	kernel = kernel - boxfilter
	sharpen_img = cv2.filter2D(img,-1,kernel)
	return sharpen_img
