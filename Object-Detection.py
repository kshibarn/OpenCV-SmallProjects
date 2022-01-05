import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.001  # Reduce value if more boundary boxes occurs

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# print(classNames)
# print(len(classNames))

'''Model Files'''
modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

'''Finding Objects Function'''


def objectFinder(outputs, img):
    ht, wt, ct = img.shape  # Height, Width, Centre
    bbox = []  # Boundary Box
    classIds = []
    confs = []  # Confidence

    for output in outputs:
        for det in output:
            scores = det[:5]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wt), int(det[3] * ht)
                x, y = int((det[0] * wt) - w / 2), int((det[1] * ht) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    '''Non Maximum Suppression(For removing duplicate boundaries)'''
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 204, 102), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 204, 102), 2)


while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    outputNames = [(layerNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    # print(net.getUnconnectedOutLayers())

    outputs = net.forward(outputNames)
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    objectFinder(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
