import cv2
import dlib
import numpy as np
import mouse
from math import hypot


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)

kernel = np.ones((9, 9), np.uint8)
def eye_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def to_np(shape, dtype="int"):
# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
# loop over the 68 facial landmarks and convert them
# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
# return the list of (x, y)-coordinates
	return coords

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (int(cx), int(cy)), 4, (0, 0, 255), 2)
        i = (cx/550)
        j = (cy/430)
        mouse.move(i*1920,j*1080)
    except:
        pass

def nothing(x):
   pass
def mid(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font =cv2.FONT_HERSHEY_PLAIN

def get_Blink_Ratio(eye_pts,facial_Landmarks):
    left_point = (facial_Landmarks.part(eye_pts[0]).x, facial_Landmarks.part(eye_pts[0]).y)
    right_point = (facial_Landmarks.part(eye_pts[3]).x, facial_Landmarks.part(eye_pts[3]).y)
    centre_top = mid(facial_Landmarks.part(eye_pts[1]), facial_Landmarks.part(eye_pts[2]))
    centre_bottom = mid(facial_Landmarks.part(eye_pts[5]), facial_Landmarks.part(eye_pts[4]))
    hor_length = hypot((left_point[0] - right_point[0]), (left_point[1]-right_point[1]))
    ver_length = hypot((centre_top[0] - centre_bottom[0]), (centre_top[1]-centre_bottom[1]))
    ratio = hor_length/ver_length
    return ratio

while(True):
    _,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    for rect in rects:
        landmarks = predictor(gray, rect)
        shape = predictor(gray, rect)
        shape = to_np(shape)
        #Detect Blinking
        left_eye = get_Blink_Ratio(left, landmarks)
        right_eye = get_Blink_Ratio(right, landmarks)
        blink_ratio = (left_eye+right_eye) /2
        if blink_ratio > 5.7:
            mouse.press()
        else:
            mouse.release()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_mask(mask, left)
        mask = eye_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid1 = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = 59
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        contouring(thresh[:, 0:mid1], mid1, img)
        contouring(thresh[:, mid1:], mid1, img, True)
        for (x, y) in shape[36:48]:
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    cv2.imshow('eyes', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break

cap.release()
cv2.destroyAllWindows()
