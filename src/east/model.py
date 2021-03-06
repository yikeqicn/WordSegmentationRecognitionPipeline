'''
east model
assumptions: 
1. one image/batch only 
2. cpu only? --YP, please modify if GPU is able to be used. slow. 
3. if vertical alignment text identified, clockwise rotation. 
not safe, a strong assumption. I don't know how to smart correct text alignment.
This alignement correction should be done before text detection probably.
'''

from imutils.object_detection import non_max_suppression
import numpy as np
#import argparse
import time
import cv2
import os
import tensorflow as tf
from glob import glob

def process(y,x,geometry,scores):
    # compute the offset factor as our resulting feature maps will
    # be 4x smaller than the input image
    (offsetX, offsetY) = (x * 4.0, y * 4.0)
    
    # extract the rotation angle for the prediction and then
    # compute the sin and cosine
    angle = geometry[y,x,4]
    cos = np.cos(angle)
    sin = np.sin(angle)
    # use the geometry volume to derive the width and height of
    # the bounding box
    h = geometry[y,x,0] + geometry[y,x,2] #??? to YP are you sure?
    w = geometry[y,x,1] + geometry[y,x,3] #??? to YP are you sure?
    
    # compute both the starting and ending (x, y)-coordinates for
    # the text prediction bounding box
    endX = int(offsetX + (cos * geometry[y,x,1]) + (sin * geometry[y,x,2]))
    endY = int(offsetY - (sin * geometry[y,x,1]) + (cos * geometry[y,x,2]))
    startX = int(endX - w)
    startY = int(endY - h)
    
    # add the bounding box coordinates and probability score to
    # our respective lists
    return (startX, startY, endX, endY) , scores[y,x]  

class east_cv2:
    def __init__(self, args,experiment=None):
        self.mW= args.east_width # model width
        self.mH= args.east_height # model heigth
        print("[INFO] loading EAST text detector...") 
        tf.reset_default_graph() 
        self.net=cv2.dnn.readNet(args.east)
        self.experiment=experiment # for debug purpose
        self.args=args
        if not os.path.exists(args.debug_folder):
            os.mkdirs(args.debug_folder)
        
    def crop(self,image_path,mini_conf=0.5):
        image=cv2.imread(image_path)
        orig = image.copy()
        (H,W)=orig.shape[:2]
        rW=W/float(self.mW)
        rH=H/float(self.mH)
        image=cv2.resize(image, (self.mW, self.mH))
        (H,W)=image.shape[:2]
        layerNames = [
           "feature_fusion/Conv_7/Sigmoid",
           "feature_fusion/concat_3"]        
        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(layerNames)
     
        # parallel graph calculation on multiple graphs might be possible above
        # if parallel, loop below, remove relevant squeeze
        scores=np.squeeze(scores)
        geometry=np.squeeze(geometry).transpose(1,2,0)
        X_,Y_=np.where(scores>mini_conf)
        candi_coordinates=list(zip(X_,Y_)) # all positive coordinates 
        #print(candi_coordinates)
        rects,confidences=zip(*map(lambda z: process(z[0],z[1],geometry,scores),candi_coordinates))
        
        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        images_rlt=[]
        boundingbox_rlt=[]        
            # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            
            # crop
            img=orig[startY:endY+1,startX:endX+1]
            # deal with vertical alignment, strong assumption
            if endY-startY>1.2*(endX-startX):
            # draw the bounding box on the image
                img=img.transpose((1,0,2))
                if startX>0.9*orig.shape[1]:
                    img=np.flip(img,axis=1)
            
            # grey and size normalize, compatible to recognition
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img=cv2.resize(img,(self.args.imgsize[0],self.args.imgsize[1]),interpolation=cv2.INTER_CUBIC)
            img=cv2.transpose(img)
            
            images_rlt.append(img)
            boundingbox_rlt.append((startX,startY,endX,endY)) 
            if self.experiment!=None:
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
                                
        if self.experiment!=None:    
            imageFile=self.args.debug_folder+'east_test.jpg'
            cv2.imwrite(imageFile,orig)
            self.experiment.log_image(imageFile)
            time.sleep(.2)
            os.remove(imageFile)
        end = time.time()
        # show timing information on text prediction
        print("[INFO] text detection took {:.6f} seconds".format(end - start))
            
        return images_rlt,boundingbox_rlt
       