#!/usr/bin/python
import os

from PIL import Image


def resize(path):
    # Single image
    # im = Image.open("/Users/DanielKong/Downloads/bill.jpg")
    # f, e = os.path.splitext("/Users/DanielKong/Downloads/bill.jpg")
    # if "jpg" not in e:
    #     print(f)
    # else:
    #     imResize = im.resize((227, 227), Image.ANTIALIAS)
    #     imResize.save(f + '.jpg', 'JPEG', quality=90)

    # multiple iamges
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path + item) and "DS_Store" not in item:
            # print(item)
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            if "jpg" not in e:
                print(f)
            else:
                imResize = im.resize((227, 227), Image.ANTIALIAS)
                imResize.save(f + '.jpg', 'JPEG', quality=90)


if __name__ == "__main__":
    resize("./Resource/cropped_227/")
    # resize("./Resource/test_images/")
