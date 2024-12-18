import argparse
from collections import deque
import cv2
import numpy as np
import torch
from src.dataset import CLASSES
from src.config import *
from src.utils import get_images, get_overlay

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Google's Quick Draw Project (https://quickdraw.withgoogle.com/#)""")
    parser.add_argument("-c", "--color", type=str, choices=["green", "blue", "red"], default="green",
                        help="Color which could be captured by camera and seen as pointer")
    parser.add_argument("-a", "--area", type=int, default=3000, help="Minimum area of captured object")
    parser.add_argument("-d", "--display", type=int, default=3, help="How long is prediction shown in second(s)")
    parser.add_argument("-s", "--canvas", type=bool, default=False, help="Display black & white canvas")
    args = parser.parse_args()
    return args

def main(opt):
    color_lower = np.array(RED_HSV_LOWER)
    color_upper = np.array(RED_HSV_UPPER)
    color_pointer = RED_RGB

    points = deque(maxlen=512)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    camera = cv2.VideoCapture(0)
    is_drawing = False
    is_shown = False

    class_images = get_images("images", CLASSES)
    predicted_class = None

    # Load model
    if torch.cuda.is_available():
        model = torch.load("./trained_models/whole_model_quickdraw")
    else:
        model = torch.load("./trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
    model.eval()

    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord(" "):
            is_drawing = not is_drawing
            if is_drawing:
                if is_shown:
                    points = deque(maxlen=512)
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                is_shown = False

        # Sau khi bạn vẽ xong, và nhấn Space để kết thúc vẽ
        if not is_drawing and not is_shown:
            if len(points):
                canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                # Blur image
                median = cv2.medianBlur(canvas_gs, 9)
                gaussian = cv2.GaussianBlur(median, (5, 5), 0)
                # Otsu's thresholding
                _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                print("Threshold Image: ", thresh)  # Kiểm tra kết quả thresholding
                contour_gs, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                if len(contour_gs):
                    contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[0]
                    print("Contour Area:", cv2.contourArea(contour))  # In ra diện tích của contour
                    if cv2.contourArea(contour) > opt.area:  # Kiểm tra diện tích
                        x, y, w, h = cv2.boundingRect(contour)
                        image = canvas_gs[y:y + h, x:x + w]
                        image = cv2.resize(image, (28, 28))
                        image = np.array(image, dtype=np.float32)[None, None, :, :]
                        image = torch.from_numpy(image)
                        logits = model(image)
                        predicted_class = torch.argmax(logits[0])  # Class with the highest probability
                        print("Predicted Class:", predicted_class)  # In ra lớp dự đoán
                        is_shown = True  # Đảm bảo kết quả sẽ hiển thị
                    else:
                        print("The object drawn is too small. Please draw a bigger one!")
                        points = deque(maxlen=512)
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # Read frame from camera
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                cv2.circle(frame, center, 5, (255, 0, 0), -1) 
            if is_drawing:
                M = cv2.moments(contour)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                points.appendleft(center)
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(canvas, points[i - 1], points[i], WHITE_RGB, 5)
                    cv2.line(frame, points[i - 1], points[i], color_pointer, 2)

        if is_shown:
            predicted_class_name = CLASSES[predicted_class]
            cv2.putText(frame, f'Predicted: {predicted_class_name}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        color_pointer, 5, cv2.LINE_AA)

            frame[5:65, 490:550] = get_overlay(frame[5:65, 490:550], class_images[predicted_class], (60, 60))

        cv2.imshow("Camera", frame)
        if opt.canvas:
            cv2.imshow("Canvas", 255 - canvas)

        if cv2.waitKey(1) == ord('s'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = get_args()
    main(opt)
