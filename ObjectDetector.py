import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture(0)
widthTarget = 320
confidenceThreshold = 0.65
nmsThreshold = 0.25
previousTime = 0

# Loading coco.names
classesFile = "coco.names"
classNames = []
with open(classesFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Loading the model (Yolo V3 - 320)
modelConfig = "Yolo/yolov3-320.cfg"
modelWeights = "Yolo/yolov3-320.weights"

net = cv.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

def detectObjects(outputs, frame):
    heightT, widthT, channelT = frame.shape
    boundingBox = []
    classIDs = []
    con = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshold:
                w, h = int(det[2] * widthT) , int(det[3] * heightT)
                x, y = int((det[0] * widthT) - w / 2), int((det[1] * heightT) - h / 2)
                boundingBox.append([x, y, w, h])
                classIDs.append(classId)
                con.append(float(confidence))

    print(len(boundingBox))
    indices = cv.dnn.NMSBoxes(boundingBox, con, confidenceThreshold, nmsThreshold)

    # Making the bounding box & displaying the class and confidence
    for i in indices:
        i = i[0]
        box = boundingBox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        x1, y1 = x + w, y + h
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Making the corner edges thicker to make it look better
        # Top Left (x, y)
        cv.line(frame, (x, y), (x + 20, y), (255, 255, 0), 7)
        cv.line(frame, (x, y), (x, y + 20), (255, 255, 0), 7)
        # Top Right (x1, y)
        cv.line(frame, (x1, y), (x1 - 20, y), (255, 255, 0), 7)
        cv.line(frame, (x1, y),
         (x1, y + 20), (255, 255, 0), 7)
        # Bottom Left (x, y1)
        cv.line(frame, (x, y1), (x + 20, y1), (255, 255, 0), 7)
        cv.line(frame, (x, y1), (x, y1 - 20), (255, 255, 0), 7)
        #  # Bottom Right (x1, y)
        cv.line(frame, (x1, y1), (x1 - 20, y1), (255, 255, 0), 7)
        cv.line(frame, (x1, y1), (x1, y1 - 20), (255, 255, 0), 7)
        
        cv.putText(frame, f"{classNames[classIDs[i]]} {int(con[i] * 100)}%", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

while True:
    successfulFrameRead, frame = cap.read()
    # Transforming the image into a blob
    blob = cv.dnn.blobFromImage(frame, 1/255, (widthTarget, widthTarget), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    detectObjects(outputs, frame)

    # Displaying the frame rate
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv.putText(frame, f"FPS: {int(fps)}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 2)

    cv.imshow("Webcam", frame)
    key = cv.waitKey(1)
    if key == 81 or key == 113:
        break
