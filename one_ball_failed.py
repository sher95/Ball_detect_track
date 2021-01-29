import numpy as np
import imutils
import cv2

colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'yellow': (0, 255, 217)}
ball = []
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    if ret is False:
        break

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    yellowLower = np.array([23, 59, 119])
    yellowUpper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, yellowLower, yellowUpper)

    (contours, hierarchy) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]
    center = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        # Print the centroid coordinates (we'll use the center of the
        # bounding box) on the image
        text = "x: " + str(x) + ", y: " + str(y)
        cv2.putText(frame, text, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(frame, center, int(radius), colors['yellow'], 2)

            ball.append(center)
        except:
            pass

        if len(ball) > 2:
            for i in range(1, len(ball)):
                cv2.line(frame, ball[i - 1], ball[i], (0, 0, 255), 4)  # harakat traektoriya

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
