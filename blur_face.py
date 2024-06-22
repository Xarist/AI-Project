import os

import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 500)
cap.set(4, 500)

eyes_cascade = cv2.CascadeClassifier('haar.xml')

frame_count = 0
output_dir = os.path.dirname(os.path.abspath(__file__))

while True:
    success, img = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=51)

    if len(eyes) > 0:
        x_min = min([x for (x, y, w, h) in eyes])
        y_min = min([y for (x, y, w, h) in eyes])
        x_max = max([x + w for (x, y, w, h) in eyes])
        y_max = max([y + h for (x, y, w, h) in eyes])

        padding = 25
        x_min = max(0, x_min - padding)
        x_max = min(img.shape[1], x_max + padding)

        combine_rect = (x_min, y_min, x_max - x_min, y_max - y_min)

        (x, y, w, h) = combine_rect
        roi_color = img[y:y + h, x:x + w]

        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

        roi_gray_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

        blured_img = cv2.GaussianBlur(roi_gray_bgr, (31, 91), 0)

        img[y:y + h, x:x + w] = blured_img

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Result', img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        frame_count += 1
        output_path = os.path.join(output_dir, f'frame_{frame_count}.png')
        cv2.imwrite(output_path, img)
        print(f'Кадр сохранен как {output_path}')


