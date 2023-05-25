import cv2
import mediapipe as mp

mpdrawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands
hands = mphands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

def drawHandLandmarks(image, handLandmarks):
    if handLandmarks:
        for lm in handLandmarks:
            mpdrawing.draw_landmarks(image, lm, mphands.HAND_CONNECTIONS)

while True:
    success, image = cap.read()
    image=cv2.flip(image,1)
    results = hands.process(image)
    handlm = results.multi_hand_landmarks
    
    drawHandLandmarks(image, handlm)

    cv2.imshow("Media Controller", image)

    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()
