import cv2

class Gender:
    def __init__(self, image):
        self.image = image


    def gender_network(self):
        self.genderlist = ['Male', 'Female']
        genderProto = r"gender_classification/deploy_gender.prototxt"
        genderModel = r"gender_classification/gender_net.caffemodel"
        gender_net = cv2.dnn.readNetFromCaffe(genderProto, genderModel)

        return gender_net


    def gender_classification(self):
        genderBlob = cv2.dnn.blobFromImage(self.image, 1, (227, 227), 127.5, swapRB=False)
        gender_net = self.gender_network()
        gender_net.setInput(genderBlob)
        genderPreds = gender_net.forward()
        gender = self.genderlist[genderPreds[0].argmax()]

        return gender