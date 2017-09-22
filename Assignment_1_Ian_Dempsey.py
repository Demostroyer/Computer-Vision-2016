# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:39:59 2016

@author: Ian Dempsey, 
@student number: 12383546
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy as sp
from scipy.spatial import distance

def fn():    
    """SO load in the data first"""
data = np.loadtxt('data.txt')
fig = plt.figure()
ax = fig.gca(projection="3d")
#accessing the first three columns. This is the 3D data
ax.plot(data[:,0], data[:,1], data[:,2],'k.')
fig = plt.figure()
#accessing the last two columns. 2D elements.
ax = fig.gca()
ax.plot(data[:,3], data[:,4],'r.')

plt.show()

def calibrateCamera3D(data):
           
    X=data[:,0]
    Y=data[:,1]
    Z=data[:,2]   
    #empty measurement Matrix
    array1=np.zeros((982,12))
    array1
    
    #filling 3D co-ords into P matrix
    n=0
    for item in range(0,982,2): 
        array1[item,0]=X[n]
        array1[item,1]=Y[n]
        array1[item,2]=Z[n]
        array1[item,3]=1
        n=n+1
    
    #array1
    n=0
    for item in range(1,982,2): 
        array1[item,4]=X[n]
        array1[item,5]=Y[n]
        array1[item,6]=Z[n]
        array1[item,7]=1
        n=n+1
    
    #array1
    #array1[980,2]
    #Z[490]
    #array1
    
    #array1[:,7]
    #Making all x1 and y1 functions negative for use in P matrix
    x1=data[:,3]
    y1=data[:,4]
    negx1=data[:,3]
    negy1=data[:,4]
    negx1
    for item in range(len(negx1)):
        negx1[item]=negx1[item]*-1
        negy1[item]=negy1[item]*-1
    negx1
    negy1
    
    #Create arrays for values of multiplication of x1 values and X/Y/Z values
    negx1ByX=negx1*X
    negx1ByY=negx1*Y
    negx1ByZ=negx1*Z
    
    #Create arrays for values of multiplication of y1 values and X/Y/Z
    negy1ByX=negy1*X
    negy1ByY=negy1*Y
    negy1ByZ=negy1*Z
    #negx1ByX[0]
    #now we need to fill in these values
    #first loop, for negx1ByX/Y/Z values
    n=0
    for item in range(0,982,2):
        array1[item,8]=negx1ByX[n]
        array1[item,9]=negx1ByY[n]
        array1[item,10]=negx1ByZ[n]
        array1[item,11]=negx1[n]
        n=n+1
    #second loop for negy1ByX/Y/Z values
    n=0
    for item in range(1,982,2):
        array1[item,8]=negy1ByX[n]
        array1[item,9]=negy1ByY[n]
        array1[item,10]=negy1ByZ[n]
        array1[item,11]=negy1[n]
        n=n+1
    
    #negx1ByX[4]
    #array1[8,8]
    #negx1ByX
    #print ("The Measurment matrix has been computed, and looks like this: " , array1)
    #create new 'array'
    a= array1
    #checking shape
    a.shape
    #getting transpose and dot product of a
    aTa= a.transpose().dot(a)
    aTa
    #checking shape
    aTa.shape
    #create the eigen vectors and values they correspond to
    d,v = np.linalg.eig(aTa)
    d
    v.shape
    
    d[1]
    #finding the position of the smallest eigen value. This is to find the correct vetor!
    small = d[0]
    position=0
    for i in range(len(d)):
        if d[i] <small:
            position=i
    position
    
    b=[]
    for n in range(len(v)):
        b.append(v[n,position])
    b
    #This is the camera matrix!
    P=np.zeros((3,4))
    n=0
    for i in range(4):
        P[0,n]=b[i]
        n=n+1
    n=0
    for i in range(4,8):
        P[1,n]=b[i]
        n=n+1
    n=0
    for i in range(8,12):
        P[2,n]=b[i]
        n=n+1
        
        
    P           
    #P = b[0:4],b[4:8],b[8:]    
    return P

#P is the Camera Matrix
P=calibrateCamera3D(data)
print ("P looks like this:",P)


#second function:
def visualiseCameraCalibration(data, P):
   #this is the 2d reproduction
   fig = plt.figure()
   ax = fig.gca()
   ax.plot(data[:,3], data[:,4],'r.') 
   plt.show()
   
   X=data[:,0]
   Y=data[:,1]
   Z=data[:,2] 

   # now do the 3D image representation
   #objpoints = (data[:,0],data[:,1],data[:,2])
   obj3D = np.ones((491,4))
   obj3D.shape
   for i in range(0,491):
       obj3D[i,0] = X[i]
       obj3D[i,1] = Y[i]
       obj3D[i,2] = Z[i]

   obj3D
   #so we now multiply the obj3D points by the P camera matrix
   obj3D_P = P.dot(obj3D.transpose()) 
   #checking the shape
   obj3D_P.shape
   #transpose it so we can plot all of it easier
   obj3D_PT=obj3D_P.transpose()
   obj3D_PT.shape
   #imgpoints = np.vstack((imgpoints,np.ones(imgpoints.shape)))
   fig = plt.figure() 
   ax = fig.gca(projection ="3d")
   #plot it all now
   ax.plot(obj3D_PT[:,0],obj3D_PT[:,1],obj3D_PT[:,2],'r')
   plt.show()

#onto the mean, distance etc:
def evaluateCameraCalibration3D(data,P):
   #print ("Hello")
   X=data[:,0]
   Y=data[:,1]
   Z=data[:,2] 
   data.shape
   #problem is they are not equal sized matrices, change the data one to be 3 cols
   data1=(data[:,0], data[:,1], data[:,2])
   
   # now do the 3D image representation
   #objpoints = (data[:,0],data[:,1],data[:,2])
   obj3D = np.ones((491,4))
   obj3D.shape
   for i in range(0,491):
       obj3D[i,0] = X[i]
       obj3D[i,1] = Y[i]
       obj3D[i,2] = Z[i]

   obj3D
   #so we now multiply the obj3D points by the P camera matrix
   obj3D_P = P.dot(obj3D.transpose()) 
   #checking the shape
   obj3D_P.shape
   #transpose it so we can plot all of it easier
   obj3D_PT=obj3D_P.transpose()
   obj3D_PT.shape   
   #calculate the distance, use the original data, with the matrix.P called obj3D
   dist = sp.spatial.distance.cdist(data1,obj3D_P,'euclidean')
   dist
                         
answer=evaluateCameraCalibration3D(data,P)
    