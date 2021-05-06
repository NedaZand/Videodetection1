import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
cv2.ocl.setUseOpenCL(False)
#from google.colab.patches import cv2_imshow
basePath = 'C:\\Users\\ASUS\\Desktop\\part5'

ESC =27
templateC = cv2.imread(basePath + "cv_cover.jpg",0)
x2=templateC

h, w = templateC.shape[::]

templateC = cv2.cvtColor(templateC, cv2.COLOR_BGR2RGB)
replace_vid = basePath + "ar_source.mp4"
file = basePath + "book.mp4"

feature_detector3 = cv2.ORB_create()
keypt_temp,features_temp = feature_detector3.detectAndCompute(templateC, None)

camera = cv2.VideoCapture(file)
ar_replace = cv2.VideoCapture(replace_vid)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_wr = cv2.VideoWriter((basePath + "output3.mp4"), fourcc, 20.0, (640,  480))


while True:
  flag, frame = camera.read()
  flag1,frame1 = ar_replace.read()
 
  if flag:
   if flag1:
      
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_cp = frame.copy()
    replace_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    replace_img = cv2.resize(replace_img,(w,replace_img.shape[1]))
    feature_detector4 = cv2.ORB_create()
    keypt_frame, features_frame = feature_detector4.detectAndCompute(frame, None)
 
    bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf1.match(features_frame,features_temp)
    rawMatches = sorted(matches, key = lambda x:x.distance)
 
    key_pt = np.float32([kp.pt for kp in keypt_temp])
    keypt_frame = np.float32([kp.pt for kp in keypt_frame])

    ptsTemp = np.float32([key_pt[m.queryIdx] for m in rawMatches])
    ptsframe = np.float32([keypt_frame[m.trainIdx] for m in rawMatches])
    (H, status) = cv2.findHomography(ptsTemp, ptsframe, cv2.RANSAC,5.0)  
    res = cv2.matchTemplate(templateC,frame,cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    pts_src = np.float32([[0, 0], [bottom_right[0], 0], [bottom_right[0], bottom_right[1]], [0, bottom_right[1]]])
    pts_dst = np.float32([[top_left[0], top_left[1]], [206, top_left[1]], [216, 215], [top_left[0], top_left[1]+w-1]])
    # pts_dst = np.float32([[top_left[0], top_left[1]], [top_left[0]-w-1,top_left[1]], [top_left[0]-w-1,top_left[1]-h-1], [top_left[0],top_left[1]-h-1]])
    hh = cv2.getPerspectiveTransform(pts_src,pts_dst)  

    warped = cv2.warpPerspective(replace_img, hh, (templateC.shape[1],templateC.shape[0]))
    warped = cv2.resize(warped, (replace_img.shape[1],replace_img.shape[0]), interpolation = cv2.INTER_AREA)
  
    _, mask = cv2.threshold(warped,1,255,cv2.THRESH_BINARY)
    maskk = cv2.bitwise_not(mask[:,:,0])
    maskk = cv2.resize(maskk, (replace_img.shape[1],replace_img.shape[0]), interpolation = cv2.INTER_AREA)
    frame = cv2.resize(frame, (replace_img.shape[1],replace_img.shape[0]), interpolation = cv2.INTER_AREA)
    targetMask = cv2.bitwise_and(frame,frame, mask=maskk)
   
 
    dst = cv2.add(targetMask, warped)
    cv2_imshow(dst)
    video_wr.write(dst)
    key = cv2.waitKey(20)                                 
    if key == ESC:
        break

cv2.destroyAllWindows()
video_wr.release()
camera.release()
ar_replace.release()