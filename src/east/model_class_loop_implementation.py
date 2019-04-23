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

class east_cv2:
    def __init__(self, args, experiment=None):
        self.mW= args.east_width # model width
        self.mH= args.east_height # model heigth
        print("[INFO] loading EAST text detector...") 
        tf.reset_default_graph() 
        self.net=cv2.dnn.readNet(args.east)
        self.experiment=experiment
        self.args=args
        if not os.path.exists(args.debug_folder):
            os.mkdirs(args.debug_folder)
        
    def crop(self,image_path,mini_conf=0.5,debug=False):
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
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
    
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < mini_conf:
                    continue
    
                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
    
                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
    
                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
    
                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
    
                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        
                # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        
        # loop over the bounding boxes
        images_rlt=[]
        boundingbox_rlt=[]
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
                img=img.transpose()
                if startX>0.9*orig.shape[1]:
                    img=np.flip(img,axis=1)
            
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
        
