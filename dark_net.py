from numpy import random
import cv2
import numpy as np
import time
from video import get_video_stream_from_camera
import pyrealsense2 as rs2
import frames
from measure import Projector
import numpy as np
from clean_depth import DepthProcesser, set_device_options
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation

CONFIDENCE_THRESHOLD=0.3

WIDTH, HEIGHT = 1280, 720
FPS = 30
DEPTH_MIN, DEPTH_MAX = 0.11, 10

INPUT_FILE='./yolo_v4/images/dog.jpg'
OUTPUT_FILE='./yolo_v4/images/predicted.jpg'


LABELS_FILE='./yolo_v4/coco.names'
CONFIG_FILE='./yolo_v4/yolov4.cfg'
WEIGHTS_FILE='./yolo_v4/yolov4.weights'

def create_yolo_model():
    labels = open(LABELS_FILE).read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln, labels

def get_objects_in_image(image, net, ln, labels):
    (H, W) = image.shape[:2]
    #start = time.time()
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD)
    boxes_coords = []
    names = []
    if len(idxs) > 0:
        
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if w > 0 and h > 0:
                start_x = max(0, x)
                start_y = max(0, y)
                end_x = min(start_x + w, image.shape[1] - 1)
                end_y = min(start_y + h, image.shape[0] - 1)
                boxes_coords.append((start_x, end_x, start_y, end_y))
                names.append(labels[classIDs[i]])
    
    return boxes_coords, names

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
    objects = []
    shapes = []
    names = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in colors[classIDs[i]]]
            #print(w, h)
            if w > 0 and h > 0:
                shapes.append((x, y, w, h))
                
                start_x = max(0, x)
                start_y = max(0, y)
                end_x = min(start_x + w, image.shape[1] - 1)
                end_y = min(start_y + h, image.shape[0] - 1)
                names.append(labels[classIDs[i]])
                objects.append(image[start_y:end_y, start_x:end_x].copy())
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            if 'bottle' in text:
                print(start_x, start_y, end_x, end_y)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    return image


if __name__ == '__main__':
    LABELS = open(LABELS_FILE).read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ln = net.getLayerNames()

    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    #net, ln, labels = create_yolo_model()
    
    pipe = rs2.pipeline()
    cfg = rs2.config()
    cfg.enable_stream(rs2.stream.depth, WIDTH, HEIGHT, rs2.format.z16, FPS)
    cfg.enable_stream(rs2.stream.color, WIDTH, HEIGHT, rs2.format.bgr8, FPS)
    profile = pipe.start(cfg)
    device = profile.get_device()
    set_device_options(profile)
    for _ in range(5):
        pipe.wait_for_frames()

    projector = Projector(device, profile, DEPTH_MIN, DEPTH_MAX)
    processer = DepthProcesser()
    n_clusters = 15
    model = KMeans(n_clusters=n_clusters)
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for i in range(n_clusters)]
    while True:
        frame, depth_frame, color_frame = frames.get_frame(pipe)
        depth_frame = processer.process(frame, False)
        depth_image, color_image = frames.to_image_representation(depth_frame=depth_frame, color_frame=color_frame)
        
        #print(color_image.shape)
        boxes, names = get_objects_in_image(color_image, net, ln, LABELS)
        
        COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

        #prediction = get_predict(color_image, LABELS, net, ln, COLORS)
        depth_data = np.asanyarray(depth_frame.get_data())


        projector.set_values(depth_frame, color_frame)

        
        cv2.imshow('depth', depth_image)

        key = cv2.waitKey(1)
        if key == 27:
            break
        pcl = np.empty(shape=(HEIGHT,WIDTH,3))
        for i in range(HEIGHT):
            for j in range(WIDTH):
                pt = projector.pixel2point((j, i))
                pcl[i,j] = pt
        print(pcl[0, 0])
        print(pcl[105, 214])

        pcl = pcl.reshape(HEIGHT*WIDTH, 3)
        print(pcl[1:3])





        
        pred = model.fit_predict(pcl)
        pred = pred.reshape(HEIGHT,WIDTH)
        print(pred)
        for y in range(len(color_image)):
            for x in range(len(color_image[0])):
                color_image[y, x] = colors[pred[y, x]]

        """for name, box in zip(names, boxes):

            
            if name != 'bottle':
                continue
            (start_x, end_x, start_y, end_y) = box 
            #print(start_x, start_y, end_x, end_y)
            depth_start = projector.color2depth((start_x, start_y)).astype(int)
            depth_end = projector.color2depth((end_x, end_y)).astype(int)
            #print(depth_start, depth_end)
            #
            try:
                object_depth = depth_data[depth_start[1]:depth_end[1], depth_start[0]:depth_end[0]]
                cv2.imshow(name, depth_image[depth_start[1]:depth_end[1], depth_start[0]:depth_end[0]])
                cv2.rectangle(color_image, (start_x,start_y), (end_x, end_y), (0, 255, 0), 2)
                pred = model.fit_predict(object_depth.flatten().reshape(-1, 1))
                pred = pred.reshape(object_depth.shape)
                for y in range(len(pred)):
                    for x in range(len(pred[0])):
                        if pred[y, x] == 0:
                            color_image[start_y + y, start_x + x] = [255, 0, 0]

            except Exception as e:
                print(e)"""
        cv2.imshow('color', color_image)

cv2.destroyAllWindows()
