#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Importing necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt


# In[18]:


# Reading images to be stitched
i01 = cv2.imread('/home/ritu/Desktop/VisualRecognition/GroupAssgnmt/img1.jpg')
i02 = cv2.imread('/home/ritu/Desktop/VisualRecognition/GroupAssgnmt/img2.jpg')
i03 = cv2.imread('/home/ritu/Desktop/VisualRecognition/GroupAssgnmt/img4.jpg')
i04 = cv2.imread('/home/ritu/Desktop/VisualRecognition/GroupAssgnmt/img5.jpg')


# In[11]:


# Reading images to be stitched
i01 = cv2.imread('/home/ritu/Desktop/VisualRecognition/GroupAssgnmt/Assignment2_2/Assignment3/Panorama/institute1.jpg')
i02 = cv2.imread('/home/ritu/Desktop/VisualRecognition/GroupAssgnmt/Assignment2_2/Assignment3/Panorama/institute2.jpg')
i03 = cv2.imread('/home/ritu/Desktop/VisualRecognition/GroupAssgnmt/Assignment2_2/Assignment3/Panorama/secondPic1.jpg')
i04 = cv2.imread('/home/ritu/Desktop/VisualRecognition/GroupAssgnmt/Assignment2_2/Assignment3/Panorama/secondPic2.jpg')


# In[19]:


# Defining a function 'trim' to remove black part obtained after stitching having area of common regions
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])    
    return frame


# In[20]:


# Defining a function 'stitch' that performs all necessary operations to get panorama image
def stitch(image01,image02,r,g,b):
    
    image1 = cv2.cvtColor(image01,cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image02,cv2.COLOR_BGR2GRAY)
    
    # Extracting features using SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Finding the key points (kp) and descriptors (des) with SIFT
    kp1, des1 = sift.detectAndCompute(image1,None)
    kp2, des2 = sift.detectAndCompute(image2,None)

    ky1 = cv2.drawKeypoints(image01,kp1,None)
    ky2 = cv2.drawKeypoints(image02,kp2,None)
    
    # BFmatcher matches the most similar features and knnMatcher with k=2 gives 2 best matches for each descriptor.
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # The features may be existing in many places of the image which can mislead the operations. 
    # So,we filter out through all the matches to obtain the best ones by applying ratio test.
    # We consider a match if the ratio defined below is predominantly greater than the specified ratio.
    good = []
    for m in matches:
        if m[0].distance < 0.55*m[1].distance:
            good.append(m[0])
            matches = np.asarray(good)
            
    draw_params = dict(matchColor = (r,g,b), 
                   singlePointColor = None,
                   flags = 2)
    match_line_1 = cv2.drawMatches(image01,kp1,image02,kp2,good,None,**draw_params)
            
    matches = bf.knnMatch(des2,des1, k=2)
    good = []
    for m in matches:
        if m[0].distance < 0.55*m[1].distance:
            good.append(m[0])
            matches = np.asarray(good)
    
    # A homography matrix is needed to perform the transformation using RANSAC and requires atleast 4 matches.
    if len(matches) >= 4:
        src = np.float32([ kp2[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst = np.float32([ kp1[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    
    else:
        print("Insufficient keypoints")
        
    # Now the images are warped and stitched together.
    dst = cv2.warpPerspective(image02,H,(image01.shape[1] + image02.shape[1], image01.shape[0]))
    dst[0:image01.shape[0], 0:image01.shape[1]] = image01
    
    # Triming and crop a little from right of stitched image to remove the undesired black portion obtained while stitching.
    dst = trim(dst)
    height = dst.shape[0]
    width = dst.shape[1] 

    y=0
    x=0
    h=int(height)
    w=int(0.97*width)
    dst = dst[y:y+h, x:x+w]
    
    return ky1,ky2,match_line_1,dst


# In[21]:


# Performing stitching operations on multiple images to create a panorama.
ky1,ky2,match_line1,i001 = stitch(i01,i02,0,255,0)
ky3,ky4,match_line2,i002 = stitch(i02,i03,255,0,0)
ky5,ky6,match_line3,i003 = stitch(i03,i04,100,100,255)

ky7,ky8,match_line4,part1 = stitch(i001,i002,0,255,0)
ky9,ky10,match_line5,final = stitch(part1,i003,0,255,0)


# In[14]:


ky1,ky2,match_line1,i001 = stitch(i01,i02,0,255,0)
ky3,ky4,match_line2,i002 = stitch(i03,i04,255,0,0)


# In[22]:


# Displaying output results of each steps of stitching operation.
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(100,100))   

data = cv2.hconcat([i01,i02,i03,i04])
keypoints = cv2.hconcat([ky1, ky2, ky4, ky6])
matches = cv2.hconcat([match_line1, match_line2, match_line3])

ax[0].imshow(data)
ax[1].imshow(keypoints)
ax[2].imshow(matches)
ax[3].imshow(final)


# In[15]:


# Displaying output results of each steps of stitching operation.
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(100,100))   

data = cv2.hconcat([i01,i02])
keypoints = cv2.hconcat([ky1, ky2])

ax[0].imshow(data)
ax[1].imshow(keypoints)
ax[2].imshow(match_line1)
ax[3].imshow(i001)


# In[16]:


# Displaying output results of each steps of stitching operation.
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(100,100))   

data = cv2.hconcat([i03,i04])
keypoints = cv2.hconcat([ky3, ky4])

ax[0].imshow(data)
ax[1].imshow(keypoints)
ax[2].imshow(match_line2)
ax[3].imshow(i002)


# In[9]:


# Displaying intermediate panoramas
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(100,100))   

ax[0].imshow(i001)
ax[1].imshow(i002)
ax[2].imshow(i003)


# In[ ]:




