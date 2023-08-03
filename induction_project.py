import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Failed to open video source.")
    exit()
cap.set(3, 1280)
cap.set(4, 720)

# Importing all images
imgBackground = cv2.imread(
    r"C:\Users\ssh12\Desktop\open cv induction\Final\COPS_background.png")
imgGameOver = cv2.imread(
    r"C:\Users\ssh12\Desktop\open cv induction\Final\Game over background.png")
imgBall = cv2.imread(
    r"C:\Users\ssh12\Desktop\open cv induction\Final\Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread(
    r"C:\Users\ssh12\Desktop\open cv induction\Final\bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread(
    r"C:\Users\ssh12\Desktop\open cv induction\Final\bat2.png", cv2.IMREAD_UNCHANGED)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ballPos = [100, 100]
speedX = 30
speedY = 30
gameOver = False
score = [0, 0]
GameKey = True
max_score = 0
while GameKey:
    # stop the game
    if cv2.waitKey(20) & 0xFF == ord('e'):
        break
    _, img = cap.read()
    if img is not None:
        img = cv2.flip(img, 1)
        imgRaw = img.copy()
    else:
        print("No video feed from the camera")

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    imgBackground = cv2.resize(imgBackground, (img.shape[1], img.shape[0]))
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # Check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1
            max_score = max(max_score, score[0]+score[1])
    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (575, 360), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (28, 30, 30), 5)
        cv2.putText(img, str(max_score).zfill(2), (590, 655), cv2.FONT_HERSHEY_COMPLEX,
                    1.5, (28, 30, 30), 5)
    # If game not over move the ball
    else:

        # Move the Ball
        if ballPos[1] >= 600 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        cv2.putText(img, str(score[0]), (300, 650),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    # img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 30
        speedY = 30
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.imread(
            r"C:\Users\LENOVO\OneDrive\Desktop\CS\AI_ML\ping pong cv\Game over background.png")
    # stop the game
    if cv2.waitKey(20) & 0xFF == ord('e'):
        GameKey = False
        break
