import requests
import base64
import json
from imutils.object_detection import non_max_suppression
SERVER_URL = 'http://localhost:8501/v1/models/east:predict'
"""
input_image = open(args.image, "rb").read()
encoded_input_string = base64.b64encode(input_image)
input_string = encoded_input_string.decode("utf-8")
instance = [{"b64": input_string}]
data = json.dumps({"instances": instance})
#########
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

# Download the image
dl_request = requests.get(IMAGE_URL, stream=True)
dl_request.raise_for_status()

# Compose a JSON Predict request (send JPEG image in base64).
jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
#predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
#predict_request = '{"instances" : [{"b64": "%s"}]}' % base64.b64encode(dl_request.content).decode()
data = json.dumps({"signature_name": "serving_default", "instances": t.tolist()})
"""
#input_image = open("./IMG_5366.jpg", "rb").read()
#encoded_input_string = base64.b64encode(input_image)
#input_string = encoded_input_string.decode("utf-8")
#instance = [{"b64": input_string}]
import cv2 as cv2
image = cv2.imread("./IMG_5366.jpg")
orig = image.copy()
(H, W) = image.shape[:2]
 
# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)
 
# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]
 
# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets

import numpy as np
import tensorflow as tf
def read_tensor_from_image_file(
    file_name, input_height=299, input_width=299, input_mean=0, input_std=255
):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    print(file_reader)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result

t = read_tensor_from_image_file("./IMG_5366.jpg", 320, 320, 0, 255)
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
data = json.dumps({"inputs": t.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/east:predict', data=data, headers=headers)
scores = np.asarray(json_response.json().get("outputs").get("scores"))
geometry = np.asarray(json_response.json().get("outputs").get("geometry"))

(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []
min_confidence = 0.5

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
        
		if scoresData[x] < min_confidence:
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
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)
 
	# draw the bounding box on the image
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
 
# show the output image
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)