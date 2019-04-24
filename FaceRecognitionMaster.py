import cv2
import sys
from FaceRecClass import WebcamFaceRecognizer

eigen_tr_data = 'recognizer/eigenTrainingData.yml'
fisher_tr_data = 'recognizer/fisherTrainingData.yml'
lbph_tr_data = 'recognizer/trainingData.yml'


def main():

    action = str(input("\nHello! Welcome to my face recognition program.\n"
                       "Which method would you like to test?\n"
                       "If you are unsure, just enter 1.\n"
                       "\n"
                       "1. EigenFaces\n"
                       "2. FisherFaces\n"
                       "3. LBPH\n"
                       "0. Exit program\n"
                       "\n"
                       "--> "))

    if action == "1":
        eigen_rec = cv2.face.EigenFaceRecognizer_create()
        prog = WebcamFaceRecognizer(eigen_tr_data, eigen_rec)
        prog.run()
    if action == "2":
        fisher_rec = cv2.face.FisherFaceRecognizer_create()
        prog = WebcamFaceRecognizer(fisher_tr_data, fisher_rec)
        prog.run()
    if action == "3":
        lbph_rec = cv2.face.LBPHFaceRecognizer_create()
        prog = WebcamFaceRecognizer(lbph_tr_data, lbph_rec)
        prog.run()
    if action == "0":
        sys.exit()
    else:
        main()

    sys.exit()


if __name__ == "__main__":
    main()
