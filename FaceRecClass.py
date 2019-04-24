import cv2
import os
import sys
import numpy as np
from PIL import Image

train_set = 'trainSet/'
user_data = 'resources/users.txt'
haar_path = '/Users/zachcapp/Documents/OpenCV/opencv-3.3.1/data/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_path)
cam = cv2.VideoCapture(0)


class WebcamFaceRecognizer(object):

    global train_set
    global user_data
    global haar_path
    global face_cascade
    global cam

    def __init__(self, unique_train_data, unique_recognizer):

        self.unique_train_data = unique_train_data
        self.unique_recognizer = unique_recognizer

    def run(self):

        self.display_menu()

    def display_menu(self):

        action = self.menu_input()

        if action == "0":
            self.cam_test()
        if action == "1":
            self.learn_face()
        if action == "2":
            self.check_id()
        if action == "3":
            sys.exit()

    @staticmethod
    def menu_input():

        accepted = ["0", "1", "2", "3"]
        prompt = str(input("0: Test camera\n1: Authorize new user\n2: Face ID!\n3: exit\n"))

        if prompt in accepted:
            return prompt
        else:
            sys.exit()

    @staticmethod
    def get_images():

        path = train_set
        image_paths = [os.path.join(path, file) for file in os.listdir(path) if file != ".DS_Store"]
        faces = []
        id_list = []

        for IP in image_paths:

            face_image = Image.open(IP).convert('L')
            face_np = np.array(face_image, dtype=np.uint8)
            temp_id = int(os.path.split(IP)[-1].split('.')[0])
            faces.append(face_np)
            id_list.append(temp_id)
            cv2.imshow("training", face_np)
            cv2.waitKey(50)

        return id_list, faces

    @staticmethod
    def format_photos():

        path = train_set
        dirs = os.listdir(path)

        for item in dirs:

            if os.path.isfile(path + item) and item != ".DS_Store":

                im = Image.open(path + item)
                f, e = os.path.splitext(path + item)
                im_resize = im.resize((250, 250), Image.ANTIALIAS)
                im_resize.save(f + '.jpg', 'JPEG', subsampling=0, quality=100)

    def store_training_data(self):

        data = self.unique_train_data
        rec = self.unique_recognizer

        path = train_set
        ids, faces = self.get_images()

        rec.train(faces, np.array(ids))
        rec.write(data)

        cv2.destroyAllWindows()

    @staticmethod
    def cam_test():
        while True:

            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=(250, 250),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                cv2.imshow('Face', frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    @staticmethod
    def learn_face():

        f = open(user_data, 'r')
        lines = f.readlines()
        if lines:
            last = lines[-1]
        last_id = int(last.split('.')[0])
        f.close()

        identifier = last_id+1
        identifier = str(identifier)

        id_name = str(input('Please enter your name:\n'))
        pic_num = 0

        f = open(user_data, 'a')
        f.write(identifier+"."+id_name+'\n')
        f.close()

        while True:

            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(250, 250),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                gray_face = cv2.resize((gray[y: y+h, x: x+w]), (250, 250))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
                pic_num += 1
                cv2.imwrite(train_set + identifier + "." + str(pic_num) + ".jpg", gray_face)
                cv2.waitKey(500)
                cv2.imshow('Video', frame)

            if pic_num > 20:
                break

        cv2.destroyAllWindows()

    def check_id(self):

        reload_prompt = str(input("Would you like to retrain recently added images?\nEnter yes or no.\n"))

        if reload_prompt.lower() == 'yes':
            self.store_training_data()
        else:
            pass

        data = self.unique_train_data
        rec = self.unique_recognizer
        rec.read(data)

        font = cv2.FONT_HERSHEY_TRIPLEX

        while True:

            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (250, 250),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                gray_face = cv2.resize((gray[y: y+h, x: x+w]), (250, 250))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
                temp_id, conf = rec.predict(gray_face)
                num = str(temp_id)
                name = None

                with open(user_data, 'r') as inp:
                    for line in inp:
                        ln = line.split('.')
                        if ln[0] == num:
                            name = ln[1]

                conf = int(conf)
                cv2.putText(frame, name+"  "+str(conf), (x+5, y+h-5), font, fontScale=1, color=(0, 0, 255), thickness=1)

            cv2.imshow('Video', frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
