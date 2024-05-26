from human_tracking.centroidtracker import CentroidTracker
from human_tracking.trackableobject import TrackableObject
from recognize_face import Face_recognize
from ultralytics import YOLO
import numpy as np
import imutils
from time import time
import dlib  # A toolkit for making real world machine learning and data analysis applications
import cv2

args = {
    "model_path": r"model_files/retrained_model/retrained_yolov8n.pt",
    "input": r"videos/input/102-in.mp4",
    "output": "videos/output/output_03.avi",
    "confidence": 0.6,
    "skip_frames": 20
}

# load our serialized model from disk
print("[INFO] loading model...")

model = YOLO(args["model_path"])
classes = model.names
device = 'cpu'
vs = cv2.VideoCapture(args["input"])


# initialize the video writer and the frame dimensions
writer = None
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=20, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
male = 0
female = 0
totaltime = 0

F1 = Face_recognize()

# loop over frames from the video stream
while True:
    # grab the next frame
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    if args["input"] is not None and frame is None:
        break

    # resize the frame and convert from BGR to RGB for dlib
    resize_frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = resize_frame.shape[:2]

    # if we are supposed to be writing a video to disk, initialize the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,(W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or (2) the correlation trackers
    status = "Waiting"
    rects = []

    model.to(device)
    list_frame = [resize_frame]


    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []


        start_time = time()
        results = model(list_frame, verbose=False)  # here only

        end_time = time()
        one_frame_time = end_time - start_time
        totaltime += one_frame_time

        x_shape, y_shape = resize_frame.shape[1], resize_frame.shape[0]
        for result in results:
            labels = result.boxes.cls
            cord = result.boxes.xyxyn
            conf = result.boxes.conf


        n = len(labels)
        for i in range(n):
            if classes[int(labels[i])] != 'person':
                continue

            row = cord[i]
            if conf[i] >= args["confidence"]:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)

                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)


                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x1, y1, x2, y2)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

            bgr = (0, 255, 0)
            cv2.rectangle(resize_frame, (startX, startY), (endX, endY), bgr, 2)

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were moving 'up' or 'down'
    cv2.line(resize_frame, (0, H // 2 + 10), (W, H // 2 + 10), (0, 255, 255), 2)     #for in

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object

                if direction > 0 and centroid[1] > H // 2 + 10:     #for in video
                    totalDown += 1
                    to.counted = True

                    # crop face from original frame, not from resize_frame
                    resize_ratio = frame.shape[1] / W
                    (cent_x, cent_y) = (centroid[0] * resize_ratio, centroid[1] * resize_ratio)
                    (s_x, e_x) = ((round(cent_x) - 120), (round(cent_x) + 90))
                    (s_y, e_y) = ((round(cent_y) - 280), (round(cent_y)))
                    cropped_frame = frame[s_y:e_y, s_x:e_x]

                    # F1 = Face_recognize(cropped_frame, resize_ratio)
                    gender = F1.face_recog(cropped_frame)

                    if gender == 'Male':
                        male += 1
                    elif gender == 'Female':
                        female += 1

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(resize_frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(resize_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("male", male),
        ("female", female),
        ("Down", totalDown),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(resize_frame, text, (340, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(resize_frame)

    # show the output frame
    cv2.imshow("Frame", resize_frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    totalFrames += 1

print(totaltime)

if writer is not None:
    writer.release()
cv2.destroyAllWindows()
