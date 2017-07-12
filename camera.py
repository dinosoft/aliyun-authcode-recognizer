import cv2
import numpy
import matplotlib.pyplot as plot

def camare_authcode_capture(debugMode=False):
    cap = cv2.VideoCapture(0)
    from main import get_auth_code_from_image

    while(1):
        # get a frame
        ret, frame = cap.read()

        frame, code = get_auth_code_from_image(frame)
        # show a frame
        if debugMode:
            cv2.imshow("capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if len(code) == 6:
                return code

    cap.release()
    cv2.destroyAllWindows()
    return None

if "__main__" == __name__:
   camare_authcode_capture(True)
