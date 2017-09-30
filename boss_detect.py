import train_boss as tb
import img_process as process
import tensorflow as tf
import cv2
import numpy as np
import time
import itchat
from itchat.content import *
cascade_path = "haarcascades/haarcascade_frontalface_default.xml"


def find_face(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
    faces = []
    if len(facerect) > 0:
        for (index, rect) in enumerate(facerect):
            x, y = rect[0:2]
            width, height = rect[2:4]
            face = img[y - 10: y + height + 20, x - 10: x + width + 20]
            face = process.resize_img(face)
            faces.append(face)
    return faces


@itchat.msg_register([TEXT])
def detect_active(msg):
    # if msg['ToUserName'] == 'filehelper':
    #     itchat.send_msg(msg['Text'], '@@0eec80947344a49518be80c966e44e9b1c270e0a1d51703ab3417ac2ad5c638d')
    # else:
    #     msg_from = msg['FromUserName']
    #     itchat.send_msg("有趣", toUserName=msg_from)
    #     print(msg_from)
    if msg['ToUserName'] == 'filehelper':
        text = msg['Text']
        if text == 'active detect':
            DETECT_FLAG = True
        else:
            DETECT_FLAG = False


if __name__ == '__main__':
    with tf.device("/gpu:7"):
        x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        drop = tf.placeholder(tf.float32)
        pre = tb.build_model(x, drop=drop)

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, "model/-9")

    itchat.auto_login(hotReload=True)
    while(1):
        time.sleep(100/1000)
        img = cv2.imread("img/detect/detect.jpg")

        try:
            faces = find_face(img)
            images = np.asarray(faces)

            imges = img.astype('float32')
            imges /= 255
            imges = img.reshape(-1, 64, 64, 3)
            if images.shape[0] > 0:
                predict = sess.run(pre, feed_dict={x: imges, drop: 1.0})
                boss = max(np.argmax(predict, 1))
                if boss == 0:
                    print("boss!")
                    itchat.send("boss!", toUserName="filehelper")
                    time.sleep(10)
                    continue
            print("nothing happened~")
            itchat.send("nothing happened", toUserName="filehelper")
        except:
            print("something error")
            itchat.send("something error", toUserName="filehelper")

