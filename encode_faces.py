# import the necessary packages
from imutils import paths
from gender_classification.gender_classification import Gender
import face_recognition
import pickle
import cv2
import os


args = {
    "dataset": "database\staff",
    "encodings": "database\encoding_hog.pickle",
    "detection_method": "hog"     #hog/cnn
}

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
knowngenders = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert image to rgb form

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])  # 以css的方式找到face 的location

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)  # encode face 去 128 readings

    gender_object = Gender(image)
    gender = gender_object.gender_classification()

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)
        knowngenders.append(gender)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames, "genders": knowngenders}
# print(data)
with open (args["encodings"], "wb") as f:
    f.write(pickle.dumps(data))

