import cv2

capture=cv2.VideoCapture(0)
capture.set(3,640)
capture.set(4,480)
classnames=[]

with open('coco.names','rt') as f:
    classnames= f.read().rstrip('\n').split('\n')

model_path='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights,model_path)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)


while True:
    success, img= capture.read()
    classIDs, confidences, bounding_box = net.detect(img, confThreshold=0.5)

    print(classIDs, bounding_box)

    if(len(classIDs) !=0):
        for classID, confidence, box in zip(classIDs.flatten(),confidences.flatten(),bounding_box):
            cv2.rectangle(img,box,color=(0,255,0),thickness=3)
            cv2.putText(img,classnames[classID-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("output",img)
    cv2.waitKey(1)