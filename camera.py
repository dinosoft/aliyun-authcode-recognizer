import cv2
import numpy
import matplotlib.pyplot as plot

cap = cv2.VideoCapture(0)
from main import get_auth_code_from_image

while(1):
    # get a frame
    ret, frame = cap.read()

    frame, _ = get_auth_code_from_image(frame)
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
