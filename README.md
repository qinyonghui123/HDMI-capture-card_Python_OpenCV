# HDMI-capture-card_Python_OpenCV

Video capture card: used to cut, capture and copy video signals, directly capture the signals of the hardware video output interface, obtain video signals from the hardware level, and then return the video signals to the system to generate video files.
We cut off the signal originally output to the display from the middle, copy one copy in the middle, and output one copy to the original display device as usual, and copy the other copy down through some special means, and then recreate the copied video signal. By encoding it into video and outputting it to the machine used to collect data, we can still record or capture video signals in real time and push the stream even when the display is working normally.
After using the USB video capture card to capture and transmit the signal to the computer, the signal can be stored with the help of capture software. Call the capture card, select the corresponding video and audio devices, and set the storage resolution, format (MJPG), bit rate, etc. according to your needs.


Python+OpenCV usage steps
1.import library
#Python 3.7.4
import cv2   #版本为4.5.2
import numpy as np

2.Real-time reading of the screen
cap0 = cv2.VideoCapture(1+ cv2.CAP_DSHOW)  
#cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  

cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
while(cap0.isOpened()):
    ret,frame=cap0.read()
    if ret==True:
        cv2.imshow("frame", frame)
    pass
    if cv2.waitKey(1000)&0xFF==ord("q"):
        break
    pass
pass
cap0.release()
cv2.destroyAllWindows()


