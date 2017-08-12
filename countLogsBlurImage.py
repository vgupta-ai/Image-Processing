import cv2
import numpy as np
from utils import *

img_org = read_image('r2_cropped.jpg')
img = improve_contrast(img_org)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = sharpen(img)
cv2.imshow('org_enhanced_gray',img)

#img = threshold(img)
#cv2.imshow('org_thresh',img)

#img = cv2.GaussianBlur(img,(5,5),2)
#cv2.imshow('blur',img)

#img = dilate(img,np.ones((5,5), np.uint8),1)
#cv2.imshow('dilation',img)

#img = opening(img)
#cv2.imshow('opening',img)

#img = closing(img)
#cv2.imshow('closing',img)

#img = get_edges(img,30,160)
#cv2.imshow('edges',img)

#img = erode(img)
#cv2.imshow('eroded',img)

#Hough Transforms
circles = find_hough_circles(img,0,50)
draw_circles(img_org,circles)

cv2.imshow('detected circles',img_org)
save_results('hough_results.jpg',img_org)

cv2.waitKey(0)
cv2.destroyAllWindows()
