

import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
cv2.namedWindow("imgwindow", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("imgwindow", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Importing all images
imgBackground = cv2.imread(r"C:\Users\LENOVO\OneDrive\Desktop\CS\AI_ML\ping pong cv\COPS_background.png")
imgGameOver = cv2.imread(r"C:\Users\LENOVO\OneDrive\Desktop\CS\AI_ML\ping pong cv\Game over background.png")
imgBall = cv2.imread(r"C:\Users\LENOVO\OneDrive\Desktop\CS\AI_ML\ping pong cv\Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread(r"C:\Users\LENOVO\OneDrive\Desktop\CS\AI_ML\ping pong cv\bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread(r"C:\Users\LENOVO\OneDrive\Desktop\CS\AI_ML\ping pong cv\bat2.png", cv2.IMREAD_UNCHANGED)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

#Variables
ballPos = [100, 100]
speedX = 30
speedY = 30
score = [0, 0]
gameOver = False
max_score=0
while True:
    success, img = cap.read()
    
    img = cv2.flip(img, 1)
     # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    if success:
        
        img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

        
         # Check for hands
        if hands:
            for hand in hands:
                x, y, w, h = hand['bbox']
                h1, w1, _ = imgBat1.shape
                y1 = y - h1 // 2
                y1 = np.clip(y1, 20, 700)
                
                
                if hand['type'] == "Left":
                    img = cvzone.overlayPNG(img, imgBat1, (50,y1))
                    if 50 < ballPos[0] < 50 + w1 and y1-h1//2 < ballPos[1] < y1 + h1//2:
                        speedX = -speedX
                        ballPos[0] += 30
                        score[0] += 1
                if hand['type'] == "Right":
                    img = cvzone.overlayPNG(img, imgBat2, (1840, y1))
                    if 1840 - 50 < ballPos[0] < 1840 and y1-h1//2 < ballPos[1] < y1 + h1//2:
                        speedX = -speedX
                        ballPos[0] -= 30
                        score[1] += 1
                max_score=max(max_score,score[0]+score[1])
        # Game Over
        if ballPos[0] < 40 or ballPos[0] > 1850:
            gameOver = True
        
        if gameOver:
            img = imgGameOver
            cv2.putText(img, str(score[1] + score[0]).zfill(2), (883, 520), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (28,30,30), 5)
            cv2.putText(img, str(max_score).zfill(2), (900, 975), cv2.FONT_HERSHEY_COMPLEX,
                    1.5, (28,30,30), 5)
        # If game not over move the ball
        else:
            # Move the Ball
            if ballPos[1] >= 800 or ballPos[1] <= 10:
                speedY = -speedY

            ballPos[0] += speedX
            ballPos[1] += speedY
            # Draw the ball
            img = cvzone.overlayPNG(img, imgBall, ballPos)
            cv2.putText(img, str(score[0]), (480, 980),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (28, 30, 30), 5)
            cv2.putText(img, str(score[1]), (1440, 980),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (28, 30, 30), 5)




        cv2.imshow("imgwindow", img)
        key = cv2.waitKey(1)
        if key==ord("e"):
            break
        elif key == ord('r'):
            ballPos = [100, 100]
            speedX = 30
            speedY = 30
            gameOver = False
            score = [0, 0]
            imgGameOver = cv2.imread(r"C:\Users\LENOVO\OneDrive\Desktop\CS\AI_ML\ping pong cv\Game over background.png")
            max_score=0
cap.release()
cv2.destroyAllWindows()      
        
