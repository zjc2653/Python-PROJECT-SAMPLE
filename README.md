# Python-PROJECT-SAMPLE
Senior research project on facial recognition algorithms using OpenCV library for Python.



Final Report (Dec. 7, 2017)

Overview

For my final project, I used OpenCV to create a facial recognition program. Facial recognition is a process involving many aspects of computer vision methods such as image processing, feature detection, object recognition, machine learning, and more. It is becoming increasingly present in technology we interact with on a daily basis and its applications to personal technology, cybersecurity, law enforcement, national defense, artificial intelligence, and even social media are expanding constantly. To implement my program, I used the Python programming language and OpenCV, the leading open source library for computer vision, image processing, and machine learning. The OpenCV library includes a modules for facial feature detection, learning, and recognition. The FaceRecognizer class provides anyone with basic programming knowledge the ability to access high level face recognition technology and start learning how to use it for their own applications. The goal of my project was to train the program with user submitted photos to identify whether or not I or someone else is an authorized user or not. I also aimed to test the accuracy and performance of the three primary recognition algorithms employed by OpenCV: Eigenfaces, Fisherfaces, and Local Binary Pattern Histograms (LBPH). 

Timeline

In the first week of diving in to this project, I began by gathering research on applications of the OpenCV library to facial recognition. I was impressed with the high level of research and development that has been done with its methods. I installed the necessary image processing packages and began creating basic test scripts to become familiar with the OpenCV features regarding object recognition and face detection. Once I could accurately detect the shape and features of a face, I began taking webcam “selfies” and learning how to them into a “recognizer” object. 

In the second week I began diving deeper into the development stage, first by learning how to train the recognizer with photos and get it to return confidence values based on how sure it was that it was my face in front of the webcam. I also started to test each of the algorithms independently, while altering values of various parameters, eventually finding that they did not have as much of an effect as I had hoped. I searched diligently to find how to increase accuracy with quantitative measures but had trouble finding much information. The next necessary step was to train the program with a lot more data.

The final stage, a duration of slightly more than a week, entailed finding a public database of photos of random people sourced publicly for facial recognition research. I had to first transform the photos to make them compatible with the algorithms used. Once I had them loaded in, I got about 6 or 7 friends to volunteer to be authorized users. Then I ran all kinds of tests under different lighting conditions, distances from the webcam, and different sized (x,y) areas of the face in order to see what worked best. The final days were spent organizing the code into a clean, efficient program, and eliminating unnecessary loops and iterations to decrease time and memory complexity.

Challenges

One of the main challenges I faced was the lack of information based on experiences others had attempting similar applications. Although OpenCV is well-documented, I found that a lot of research is not public since it is being used to develop state-of-the-art commercial technology. There were quite a few others posting questions on the topic and not getting answers. Another obstacle was that there was more information pertaining to development in C++ since that is the native language of OpenCV, and I was only experienced enough in Python to develop and test a program confidently and in a timely manner. 

It became clear about halfway through testing various conditions that lighting and illumination of the face was most the important factor in training datasets. Any sort of shadow from the angle of lights in the room would affect results, so the ideal way to achieve accuracy is to normalize light conditions. Orientation of the face was important as well. 
The final challenge that made extensive testing difficult was the memory complexity of the machine learning type training algorithms. The (.yml) files produced by the trainer were up to a gigabyte large each, for one database and around 100 of my own photos. To increase accuracy meant largely increasing datasets, which would have increased computing times and memory requirements significantly. If I were to go forward with the development of this program, I would attempt to do the following: acquire more relevant guidance information for developing with the FaceRecognizer class, normalize lighting and orientation of face, and reduce size of images while maintain enough information to achieve accurate matches.


Program Demonstration

https://youtu.be/Usy8zCUTLCE
