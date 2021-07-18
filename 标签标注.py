import glob
import xml.etree.ElementTree as ET
import numpy as np
import os
from PIL import Image,ImageDraw,ImageFont

path = r'Z:\data\VOCdevkit\VOC2007\Annotations'
img_path = 'Z:\data\VOCdevkit\VOC2007\JPEGImages'
save_path = r"Z:\tamppic"

def load_dataset(path,img_path,save_path):
    c = {'嘴':'red','右手':'black','左手':'pink','眼':'yellow','脸':'green'}
    for xml_file in glob.glob("{}/*xml".format(path)):
        num = xml_file.strip().split("\\")[-1][:-4]
        tree = ET.parse(xml_file)
        with Image.open(os.path.join(img_path,f"{num}.jpg")) as img:
            draw = ImageDraw.Draw(img)
            setFont = ImageFont.truetype('C:/windows/fonts/Dengl.ttf', 3)
            img_save = os.path.join(save_path,f"{num}.jpg")
            try:
                for obj in tree.iter("item"):
                    name = obj.findtext("name")
                    xmin = int(obj.findtext("bndbox/xmin"))
                    ymin = int(obj.findtext("bndbox/ymin"))
                    xmax = int(obj.findtext("bndbox/xmax"))
                    ymax = int(obj.findtext("bndbox/ymax"))
                    draw.rectangle((xmin,ymin,xmax,ymax),width=2,outline=c[name])
                    draw.text((xmin,ymin),f"{name}",font=setFont,fill=c[name])
                    img.save(img_save)
            except:
                print(img_save)

load_dataset(path,img_path,save_path)
