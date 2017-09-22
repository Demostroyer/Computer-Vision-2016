# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:20:25 2016

@author: Ian Dempsey, Student NUmber: 12383546
"""

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d
#import scipy as sp
import numpy as np
import cv2
import glob
import ipdb
#Global variables used in multiple functions
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
corners=[]
corners=np.asarray(corners,dtype="float32")

def undistort_():

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob('board*.jpg')
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    #ipdb.set_trace()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    img = cv2.imread('board7.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
#save our undistorted image
    cv2.imwrite('calibresult.png',dst)
    

def detect_():
    #these will be different as they are based off of an undistorted image
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    

    #this is the undistorted image, output from above function
    #images = glob.glob('calibresult.png')
    images = cv2.imread('calibresult.png')
    #our image to be imposed on the checkerboard, call it 'src' for short
    src = cv2.imread("trolltunga.jpg")

    #loop through and detect the corners on our checkerboard
    for i in range(0,1):
        
        gray = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)
    
    #this checks that corners is not empty
    #print(corners)    
    
    #we have corners of one of the images we will use, the undistorted one
    #we need the corners of our image to be imposed
    #lets find the height and width of our image
    height, width, channels = src.shape
    #making sure that our values are float32, as everything is in this type
    height=np.float32(height)
    width=np.float32(width)
    #print("here",type(height))
    #create array to store the points on our src image
    p = []
    #the dividers to 
    heightdiver = 0.0
    widthdiver=0.0
    #dividing the image into parts, where the two cross, they will be the points in
    #our image and they correspond to points on chessboard.
    for y in range(0,6):
         widthdiver=0.0
         for x in range(0,9):
             #appending the points into our array
             p.append([[(width*widthdiver),(height*heightdiver)]])
             widthdiver+=0.125
         heightdiver+=0.2
         
         
    # make sure that it is an array, and they are the same type     
    p=np.asarray(p,dtype=corners.dtype)

    #now we have the src points from our image
    #we will get the homography:   
    h, mask = cv2.findHomography(corners,p)
    #inversing just to correct order
    h=np.linalg.inv(h)
    
    #FINALLY h WORKS!!!!
    #now we get the src image, in the correct perspective and size
    im_out = cv2.warpPerspective(src, h, (images.shape[1],images.shape[0]))
    
    #cv2.imshow("Source Image", src)#works
    #cv2.imshow("Warped Image",im_out)

    #im_dst = images + im_out
    #i want to put the perspectivelly correct image, so i use the size of it
    rows,cols,chans =im_out.shape 
    roi = images[0:rows,0:cols] 
    #convert it to gray scale
    graysrc = cv2.cvtColor(im_out,cv2.COLOR_BGR2GRAY)
    #creating the mask, and the it's inverse
    ret,mask = cv2.threshold(graysrc, 10, 255 , cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    #black out the area in ROI, this is where the imposed image would go
    img1_bg  = cv2.bitwise_and(roi,roi,mask = mask_inv)
    #i take only the region of our 'src' image
    img2_fg = cv2.bitwise_and(im_out,im_out,mask=mask)
    #puts the two together and the modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    images[0:rows,0:cols] = dst
    #displays result
    cv2.imshow('result',images)
    
#call first function
undistort_()
#call second function
detect_()