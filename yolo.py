# USAGE
# python yolo.py --image "images/imageName.jpg" --yolo yolo-coco
# python yolo.py --folder "Folder" --yolo yolo-coco

#remove # from lines 163/164 to pop image with bbox's on the original image

# import
import numpy as np
import argparse
import time
import cv2
import os
import glob
import wget
import json
from json.decoder import JSONDecodeError

# get weights from an existing model
if not os.path.isfile('yolo-coco/yolov3.weights'):
	url = 'https://pjreddie.com/media/files/yolov3.weights'
	wget.download(url,'yolo-coco')



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
ap.add_argument("-f", "--folder",required=False,
	help="path to input images folder")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())




# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent a class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# the paths to the YOLO weights and model cfg
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# YOLO object detector trained on COCO dataset
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

blob =''

# determine the output layer names that we need from YOLO (classes)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

(H, W) = ('','')


def getResults(blob, ln,filename):
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()


	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))


	# initialize our lists of detected bounding boxes, confidences, and classIDs
	boxes = []
	confidences = []
	classIDs = []


	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each detection
		for detection in output:
			# extract the classID and confidence of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out predictions by ensuring the detection probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to the
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,and classIDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression on BBOX's
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
							args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		objectDict = {}
		count = 0
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			objectLabel = "{}".format(LABELS[classIDs[i]])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
						0.5, color, 2)

			# objectBbox = (objectLabel, ((x, y), (x + w, y + h)))
			objectBbox = {}
			objectLabel = LABELS[classIDs[i]]
			if objectDict:
				if objectLabel in objectDict[filename]:
					count = count + 1
					objectLabel = objectLabel + " {}".format(count)
				objectBbox[objectLabel] = ((x, y), (x + w, y + h))
				objectDict[filename].update(objectBbox)
			else:
				objectBbox[objectLabel] = ((x, y), (x + w, y + h))
				objectDict[filename] = objectBbox


		# encode data unto json file
		data = {}
		with open('BBOXcoords.json','a+') as fp:
			try:
				fp.seek(0)
				data = json.load(fp)
			except JSONDecodeError:
				pass

			if filename in data:
				print("image with name",'"' , filename,'"' ,"was already scanned.")
				pass
			else:

				data.update(objectDict)
				with open('BBOXcoords.json','w'):
					json.dump(data,fp)
			fp.close()

	#show the output image
	# cv2.imshow("image", image)
	# cv2.waitKey(0)

# load our input image and save dimensions
if args['image'] != None:
	try:
		cv2.namedWindow("image", cv2.WINDOW_NORMAL)
		image = cv2.imread(args["image"])
		image = cv2.resize(image,(480,480))
		(H, W) = image.shape[:2]
	except AttributeError:
		print("cannot load file named: ", args["image"])
		print("Please check if the file exists, if it is: check the file name format to end with .jpg")

	# construct a blob from the input image and then do a forward pass
	blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(512, 512),
		swapRB=True, crop=False)
	getResults(blob,ln,args["image"])
elif args['folder'] != None :			#else a folder was entered as input
	for file in glob.glob(args['folder']+'/*'):
		print('image to detect: ',file)
		if file.endswith(".jpg"):
			cv2.namedWindow("image", cv2.WINDOW_NORMAL)
			image = cv2.imread(file)
			image = cv2.resize(image, (480, 480))
			(H, W) = image.shape[:2]
			# construct a blob from the input image and then do a forward pass
			blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(448, 448),
			swapRB=True, crop=False)
			getResults(blob,ln,file)
