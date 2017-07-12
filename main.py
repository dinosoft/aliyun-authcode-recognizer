import cv
import cv2
import numpy as np

from knn import knn_clf


def grayify(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(img, 5)
    return img


def thresholding_inv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, 100, 255, cv.CV_THRESH_BINARY_INV)
    # bin = cv2.medianBlur(bin, 3)

    return bin


def rotate180(image):
    img = cv2.flip(image, -1)
    return img


def rotate90(image):
    img = cv2.transpose(image)
    img = cv2.flip(img, 0)
    return img


def find_auth_code_img_rec(contours, image_area):
    pre_cnt = None
    auth_code_img = []

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        candidates.append([x, y, w, h])

    candidates = sorted(candidates, key=lambda x: x[0] * 10000 + x[1])

    for cad in candidates:
        x, y, w, h = cad
        if image_area * 0.015 > w * h > image_area * 0.0001 and w > 10 and h > 20 and h > 1.2 * w:
            if pre_cnt is None:
                pre_cnt = [x, y, w, h]
                auth_code_img.append(pre_cnt)
                continue

            x2, y2, w2, h2 = pre_cnt
            pre_cnt = [x, y, w, h]
            if abs(x - x2) + abs(y - y2) < w + h and max(w * h, w2 * h2) / min(w * h, w2 * h2) < 5:
                auth_code_img.append(pre_cnt)
                if len(auth_code_img) == 6:
                    return auth_code_img
            else:
                auth_code_img = [pre_cnt]

    return auth_code_img


def get_auth_code_from_image(image):
    # image = cv2.imread("D:/camera_digit/b.png")
    # image = thresholding_inv(image)
    image = rotate180(image)
    gray_img = grayify(image)

    th_val, image_th = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
    # image_th = cv2.medianBlur(image_th, 3)

    # revert the image
    image_th = 255 - image_th

    contours, hierarchy = cv2.findContours( image_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]

    auth_code_img_rec = find_auth_code_img_rec(contours, image_area)

    recognizer = knn_clf()

    index = 0
    auth_code_text = ''

    cv2.drawContours(image, contours, -1, (255,0, 0), 3)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    import matplotlib.pyplot as plt

    if auth_code_img_rec is not None:
        for rec in auth_code_img_rec:
            index += 1
            [x, y, w, h] = rec
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if w > 20:
                kernel = np.ones((2, 2), np.uint8)
                image_th2 = cv2.dilate(image_th, kernel, iterations=1)
            else:
                image_th2 = image_th

            cropped = image_th2[y: y + h, x:x + w]
            BORDER_SIZE = w/5
            cropped = cv2.copyMakeBorder(cropped, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                                         cv2.BORDER_CONSTANT, (255, 255, 255))
            resized = cv2.resize(cropped, (15, 20), interpolation=cv2.INTER_AREA)

            # convert to float
            fea = resized / 1.0

            if h > 2.2*w:
                digit_text = 1
                auth_code_text = auth_code_text + '1'
            else:
                digit_text = recognizer.predict_digit(fea)
                auth_code_text = auth_code_text + str(int(digit_text))

            plt.subplot(1, 6, index)
            plt.axis('off')
            plt.imshow(cropped, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title("#{}".format(digit_text) )
    plt.show()

    if auth_code_img_rec is not None and len(auth_code_img_rec)>1:
        [x, y, _, _] = auth_code_img_rec[0]
        y -= 50
        cv2.putText(image, auth_code_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    return image, auth_code_text
