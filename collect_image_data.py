import os
import cv2
cap=cv2.VideoCapture(0)
directory= os.path.join('Image/')
actions = ["peace", "thumbs up", "stop"]
for action in actions:
    try:
        os.makedirs(os.path.join(directory, action))
    except:
        pass
while True:
    _,frame=cap.read()

    # display number of images in each folder
    count = {
             'peace': len(os.listdir(directory+"peace")),
             'thumbs up': len(os.listdir(directory+"thumbs up")),
             'stop': len(os.listdir(directory+"stop")),
               }
    cv2.putText(frame, "peace : "+str(count['peace']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "thumbs up : "+str(count['thumbs up']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "stop : "+str(count['stop']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

    row = frame.shape[1]
    col = frame.shape[0]

    #create a rectangular shape where user places the hand 
    cv2.rectangle(frame, (0, 40), (300, 400), (0, 255, 0), 2)

    cv2.imshow("data",frame)
    cv2.imshow("ROI",frame[40:400,0:300])
    frame=frame[40:400,0:300]
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory+'peace/'+str(count['peace'])+'.png',frame)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'thumbs up/'+str(count['thumbs up'])+'.png',frame)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'stop/'+str(count['stop'])+'.png',frame)
    if interrupt == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()