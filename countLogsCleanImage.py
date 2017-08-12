import cv2
import numpy as np
from utils import *

img_org = read_image('AR9_cropped.jpg')
img = cv2.cvtColor(img_org,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img,(7,7),2)

#Hough Transforms
circles = find_hough_circles(img,0,0)
draw_circles(img_org,circles)

cv2.imshow('detected circles',img_org)
save_results('hough_results.jpg',img_org)

cv2.waitKey(0)
cv2.destroyAllWindows()
