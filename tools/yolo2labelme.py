import json
import glob
import argparse
import os
import copy
import tqdm
from PIL import Image
import numpy as np



class Labelme:
    def __init__(self, image_dir: str, yolo_dir: str):
        self.images_dir = image_dir
        self.yolo_dir = yolo_dir
        for index, (im, yo) in enumerate(zip(image_dir, yolo_dir)):
            if im != yo:
                break
        self.real_path = os.path.dirname(image_dir[:index])
        self.image_real_path = os.path.relpath(image_dir, self.real_path)
        self.txt_real_path = os.path.relpath(yolo_dir, self.real_path)
        self.labelme = {
            "version": "5.4.1", "flags": {},
            "shapes": [], "imagePath": "",
            "imageData": None,
            "imageHeight": 0, "imageWidth": 0}
        self.shape = {
            "label": "white","points": [],
            "group_id": None,"description": "",
            "shape_type": "rectangle", "flags": {},
            "mask": None}

    def __call__(self, *arg):
        self.new_labelme = None
        image = check_image_content(arg[0])
        if not image:
            return 0
        info = open(arg[1], encoding='utf-8').readlines()
        # new_info = []
        # for line in info:
        #     if 1 <= int(line.split()[0]) <= 2:new_info.append(line)
        # info = new_info
        if len(info) == 0:
            return 0
        self.new_labelme = copy.deepcopy(self.labelme)
        self.new_labelme["imagePath"] = os.path.basename(arg[0])
        self.new_labelme["imageHeight"] = image.height
        self.new_labelme["imageWidth"] = image.width
        for index, line in enumerate(info):
            label = [float(i) for i in line[:-1].split()]

            if label[-1] == 0 or label[-2] == 0: continue
            # if int(label[0]) < 1 or int(label[0]) > 2: continue

            shape = copy.deepcopy(self.shape)
            shape["label"] = f'{int(label[0])}'
            xywh = np.array(label[1:]).reshape(2, 2) * [image.width, image.height]
            shape["points"] = (xywh[0, :] - xywh[1, :] / 2).tolist(), (xywh[0, :] + xywh[1, :] / 2).tolist()
            self.new_labelme["shapes"].append(shape)

    def save(self, file_name: str):
        if self.new_labelme is None:
            return 0
        json.dump(self.new_labelme, open(file_name, 'w', encoding='utf-8'), indent=4)


def check_image_content(filename):
    try:
        image = Image.open(filename)
        image.verify()
        return image
    except (IOError, SyntaxError) as e:
        return False


def get_args():
    parser = argparse.ArgumentParser(description="seamlessClone")


    parser.add_argument("--img_dir", type=str,
                        default=rf'R:\disk1\01.Datasets\06.jz\一丰数据\907-燃油管-res\二线\0',
                        help="图片路径")
    parser.add_argument("--yolo_dir", type=str,
                        default=fr'R:\disk1\01.Datasets\06.jz\一丰数据\907-燃油管-res\二线\0-yolo-txt\labels',
                        help="txt文件路径")


    return parser.parse_args()

'''
    脚本：yolo转labelme格式
    需要修改的路径:
        img_dir
        yolo_dir
'''
if __name__ == '__main__':
    args = get_args()
    labelme = Labelme(args.img_dir, args.yolo_dir)

    # image_files = glob.glob(args.img_dir + "/**/*.*", recursive=True)
    # yolo_files = glob.glob(args.yolo_dir + "/**/*.*", recursive=True)


    image_files = []
    yolo_files = []
    for img_file in os.listdir(args.img_dir):
        if img_file.endswith(".jpg"):
            image_files.append(os.path.join(args.img_dir,img_file))
    for txt_file in os.listdir(args.yolo_dir):
        if txt_file.endswith(".txt"):
            yolo_files.append(os.path.join(args.yolo_dir,txt_file))



    for image_file in tqdm.tqdm(image_files):
        # yolo_file = os.path.splitext(image_file.replace(labelme.image_real_path, labelme.txt_real_path))[0] + '.txt'
        yolo_file = os.path.join(args.yolo_dir , os.path.splitext(image_file.split("\\")[-1])[0] + '.txt')
        if yolo_file not in yolo_files:
            print(os.path.splitext(image_file.split("\\")[-1])[0] + '.txt')
            continue
        res = labelme(image_file, yolo_file)
        labelme_file = os.path.splitext(image_file)[0] + '.json'
        if res!=0:
            labelme.save(labelme_file)