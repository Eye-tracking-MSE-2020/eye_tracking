import numpy as np
import cv2

# 얼굴과 눈을 검출하기 위해 미리 학습시켜 놓은 XML 포맷으로 저장된 분류기를 로드합니다.

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

# 얼굴과 눈을 검출할 그레이스케일 이미지를 준비해놓습니다.

while 1:

    ret, img = cap.read()

    col = img

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 이미지에서 얼굴을 검출합니다.

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 얼굴이 검출되었다면 얼굴 위치에 대한 좌표 정보를 리턴받습니다.

    for (x, y, w, h) in faces:

        # 원본 이미지에 얼굴의 위치를 표시합니다.

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 눈 검출은 얼굴이 검출된 영역 내부에서만 진행하기 위해 ROI를 생성합니다.

        roi_gray_l = gray[y:int(y + h * .6), int(x + w * .5):x + w]
        roi_gray_r = gray[y:int(y + h * .6), x:int(x + w * .5)]

        roi_color_l = img[y:int(y + h * .6), int(x + w * .5):x + w]
        roi_color_r = img[y:int(y + h * .6), x:int(x + w * .5)]

        # 눈을 검출합니다.

        eyes_r = eye_cascade.detectMultiScale(roi_gray_r, 1.3, 5)
        eyes_l = eye_cascade.detectMultiScale(roi_gray_l, 1.3, 5)

        # 눈이 검출되었다면 눈 위치에 대한 좌표 정보를 리턴받습니다.

        for (ex, ey, ew, eh) in eyes_r:
            cv2.rectangle(roi_color_r, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)  # draw rectangle around eyes
            cv2.line(roi_color_r, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)  # draw cross
            cv2.line(roi_color_r, (ex + ew, ey), (ex, ey + eh), (0, 0, 255), 1)
            pupilFrame_r = cv2.equalizeHist(roi_gray_r[ey:ey+eh, ex:ex+ew])  # using histogram equalization of better image.
            roi_color_r = roi_color_r[ey:ey+eh, ex:ex+ew]
            cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # set grid size
            clahe_r = cl1.apply(pupilFrame_r)  # clahe
            blur_r = cv2.medianBlur(clahe_r, 5)  # median blur
            circles_r = cv2.HoughCircles(blur_r, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=35, minRadius=0, maxRadius=30)  # houghcircles
            if circles_r is not None:  # if atleast 1 is detected
                circles = np.round(circles_r[0, :]).astype("int")  # change float to integer
                print(circles)
                for (cx, cy, cr) in circles:
                    cv2.circle(roi_color_r, (cx, cy), cr, (0, 255, 255), 2)
                    cv2.rectangle(roi_color_r, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 128, 255), -1)

        for (ex, ey, ew, eh) in eyes_l:
            cv2.rectangle(roi_color_l, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)  # draw rectangle around eyes
            cv2.line(roi_color_l, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)  # draw cross
            cv2.line(roi_color_l, (ex + ew, ey), (ex, ey + eh), (0, 0, 255), 1)
            pupilFrame_l = cv2.equalizeHist(roi_gray_l[ey:ey+eh, ex:ex+ew])
            roi_color_l = roi_color_l[ey:ey+eh, ex:ex+ew]
            cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # set grid size
            clahe_l = cl1.apply(pupilFrame_l)  # clahe
            blur_l = cv2.medianBlur(clahe_l, 5)  # median blur
            circles_l = cv2.HoughCircles(blur_l, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=35, minRadius=0, maxRadius=30)
            if circles_l is not None:  # if atleast 1 is detected
                circles = np.round(circles_l[0, :]).astype("int")  # change float to integer
                print(circles)
                for (cx, cy, cr) in circles:
                    cv2.circle(roi_color_l, (cx, cy), cr, (0, 255, 255), 2)
                    cv2.rectangle(roi_color_l, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 128, 255), -1)

        # 원본 이미지에 얼굴의 위치를 표시합니다. ROI에 표시하면 원본 이미지에도 표시됩니다.

        #cv2.rectangle(roi_color, (ex, ey + (int)(eh * 0.2)), (ex + ew, ey + eh - (int)(eh * 0.2)), (0, 255, 0), 2)

    # 얼굴과 눈 검출 결과를 화면에 보여줍니다.

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break