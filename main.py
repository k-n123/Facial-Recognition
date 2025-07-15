import cv2
import numpy as np
import face_recognition

# Load image file
chrisEvansImg = face_recognition.load_image_file("ImagesMain/Chris Evans.jpg")

#Convert image to RGB
chrisEvansImg = cv2.cvtColor(chrisEvansImg,cv2.COLOR_BGR2RGB)



# Load image file
chrisEvansTest = face_recognition.load_image_file("ImagesMain/Chris Test.jpg")

#Convert image to RGB
chrisEvansTest = cv2.cvtColor(chrisEvansTest,cv2.COLOR_BGR2RGB)

# Load image file
rdjTest = face_recognition.load_image_file("ImagesMain/RDJ.jpeg")

#Convert image to RGB
rdjTest = cv2.cvtColor(rdjTest,cv2.COLOR_BGR2RGB)



#Set Face location
faceLoc = face_recognition.face_locations(chrisEvansImg)[0]
#Encode face
encodeChris = face_recognition.face_encodings(chrisEvansImg)[0]

#Encode test image
encodeTest = face_recognition.face_encodings(rdjTest)[0]

#Compare fafces and print results
results = face_recognition.compare_faces([encodeChris], encodeTest)
print(results)

