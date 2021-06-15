import cv2
import numpy as np
import time
from video import get_video_stream_from_camera

CONFIDENCE_THRESHOLD=0.3

INPUT_FILE='./yolo_v4/images/dog.jpg'
OUTPUT_FILE='./yolo_v4/images/predicted.jpg'


LABELS_FILE='./yolo_v4/coco.names'
CONFIG_FILE='./yolo_v4/yolov4.cfg'
WEIGHTS_FILE='./yolo_v4/yolov4.weights'



def get_predict(image, labels, net, ln, colors):    
    (H, W) = image.shape[:2]
    #start = time.time()
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    #end = time.time()
    
#print("[INFO] YOLO took {:.6f} seconds".format(end - start))


    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections

        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE_THRESHOLD:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in colors[classIDs[i]]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)

    
    return image


if __name__ == '__main__':
    np.random.seed(4)
    LABELS = open(LABELS_FILE).read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    ln = net.getLayerNames()

    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")

    for frame in get_video_stream_from_camera():
        prediction = get_predict(frame, LABELS, net, ln, COLORS)
        cv2.imshow('Predicted', prediction[:,:,[2,1,0]])
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
