
import sys

from numpy.lib.npyio import save
import rospy
import cv2 as cv
import numpy as np
import sensor_msgs.msg

class ImageSaver():

    def __init__(self, filename):
        self._filename = filename
        self._saved_image = False
        rospy.init_node("image_saver", anonymous=False)
        rospy.Subscriber("/drone/out/image_rect_color", sensor_msgs.msg.Image, self._image_cb)

    def save_image(self):
        while not rospy.is_shutdown():
            if self._saved_image:
                print("Image saved")
                sys.exit(0)

    def _image_cb(self, msg: sensor_msgs.msg.Image):
        print("Saving image")
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        cv.imwrite(f"{self._filename}.png", img)
        self._saved_image = True

def main():
    args = sys.argv

    if len(args) != 2:
        print("Usage: save_image <filename>")
        sys.exit(1)

    saver = ImageSaver(args[1])
    saver.save_image()

if __name__ == "__main__":
    main()