import cv2
import os

cascade_path = "haarcascades/haarcascade_frontalface_default.xml"

img_path = "img/process/unchoose/"
IMAGE_SIZE = 64

def find_face(img, file_path):
    args = file_path.split('/')
    face_name = args[-2]
    file_name = args[-1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
    if os.path.exists(img_path + face_name):
        pass
    else:
        os.mkdir(img_path + face_name)
    if len(facerect) > 0:
        print(file_path)
        for (index, rect) in enumerate(facerect):
            x, y = rect[0:2]
            width, height = rect[2:4]
            face = img[y: y + height, x: x + width]
            face = resize_img(face)
            cv2.imwrite(img_path + face_name + "/" + face_name + str(index) + file_name, face)


def resize_img(img, height = IMAGE_SIZE, width = IMAGE_SIZE):
    img = draw_box(img)
    resized_img = cv2.resize(img, (height, width))
    return resized_img


def draw_box(img):
    h, w, _ = img.shape
    longest_edge = max(h, w)
    top, bottom, left, right = (0, h, 0, w)
    if w < longest_edge:
        dh = longest_edge - w
        top = dh // 2
        bottom = h - top
    elif h < longest_edge:
        dw = longest_edge - h
        left = dw // 2
        right = w - left
    else:
        pass
    BLACK = [0, 0, 0]
    img_box = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return img_box


def process_img(dir):
    for file_or_dir in os.listdir(dir):
        abs_path = os.path.abspath(os.path.join(dir, file_or_dir))
        if os.path.isdir(abs_path):
            process_img(abs_path)
        else:
            abs_path = os.path.abspath(os.path.join(dir, file_or_dir))
            if file_or_dir.endswith('.jpg'):
                img = cv2.imread(abs_path)
                try:
                    find_face(img, abs_path)
                except:
                    print(abs_path)

if __name__ == '__main__':
    path = "/data/zonghua/boss_detect/img/undo"
    process_img(path)
