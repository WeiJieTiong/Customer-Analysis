import face_recognition
import pickle
import cv2
from gender_classification.gender_classification import Gender
import os

class Face_recognize:

    def __init__(self):
        with open("database\encoding_hog.pickle", "rb") as f:
            data = pickle.load(f)

        #extract the last number of new customer to prevent overwrite of the customer data
        for name in data.get('names'):
            if 'newcustomer' in name:
                j = name.split()[1]
                self.j = int(j) + 1  # record new customer sequence
            else:
                self.j = 1


        path = os.path.join(os.path.dirname(__file__), f'database\Staff\\Unknown')
        if os.path.exists(path):
            files = os.listdir(path)[-1]     #take the last unknown name
            k = files.split()[1]             #extract the sequence number
            self.k = int(k) +1            # record unknown sequence
        else:
            self.k = 1

    def face_recog(self,cropped_frame):
        data = pickle.loads(open("database\encoding_hog.pickle", "rb").read())
        rgb_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_image ,model= 'hog')        #top, right, bottom, left
        encodings = face_recognition.face_encodings(rgb_image , boxes)
        names = []

        if encodings:       #if gt face in the frame
            # loop over the facial embeddings
            for i,encoding in enumerate(encodings):
                matches = face_recognition.compare_faces(data["encodings"],encoding)           #compare faces with database and return a list of true/false

                if True in matches:          #if gt face and recognise in database
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = data["names"][i]                           #每一个在database里 encode face 的都有名的， 从那里拿回名罢了
                        counts[name] = counts.get(name, 0) + 1            #exp. count = {'owen_grady' : 1}    0是default value in case 找不到
                    name = max(counts, key=counts.get)

                    for i, search_name in enumerate(data.get('names')):            #找到跟recognize的名字 一样
                        if search_name == name:
                            gender = data["genders"][i]             # 根据名字的顺序拿到gender


                else:                       #if gt face but dont recognise in database
                    name = 'newcustomer' + f' { self.j}'
                    # print( self.j)
                    gender_object = Gender(rgb_image)                #传多一点details 那张照片 rather than 只有脸
                    gender = gender_object.gender_classification()
                    self.j += 1                 #for next customer

                    #write face into pickle database

                    data.get("encodings").append(encoding)
                    data.get("names").append(name)
                    data.get("genders").append(gender)

                    with open("database\encoding_hog.pickle", "wb") as f:
                        f.write(pickle.dumps(data))

                    #save face image into database
                    path = os.path.join(os.path.dirname(__file__), f'database\Staff\\newCustomer\{name}')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    cv2.imwrite(f'{path}\\{name}.png',cropped_frame)

                names.append(name)


        else:  #if no face in found in image
            gender_object = Gender(rgb_image)
            gender = gender_object.gender_classification()
            name = "unknown" + f' { self.k}'                                            #if dont recognise in database, it is unknown
            names.append(name)
            self.k +=1

            # save face image into database
            path = os.path.join(os.path.dirname(__file__), f'database\Staff\\Unknown\{name}')
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(f'{path}\\{name}.png', cropped_frame)
        print(f"name:{name}, gender : {gender}")


        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):

            #cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.rectangle(cropped_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(cropped_frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        # show the output image
        cv2.imshow("Image", cropped_frame)

        return gender