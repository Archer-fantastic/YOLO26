import math
import os
import shutil

import numpy as np
from sonic.utils_func import glob_extensions

# from mmdet.utils import get_root_logger
from sonic_core.sonic_ai.base.utils.utils_load_json import load_data
from tqdm import tqdm


def shape2points(shape, height, width):
    # logger = get_root_logger()

    shape_type = shape["shape_type"]
    points = shape["points"]

    new_points = []
    if shape_type == "polygon":
        for p in points:
            # if len(points) < 3:
            #     new_points = []
            # logger.warning(f'polygon 异常，少于三个点：{shape}')
            p[0] = np.clip(p[0], 0, width)
            p[1] = np.clip(p[1], 0, height)
            new_point = (p[0] / width, p[1] / height)
            new_points.append(new_point)
    elif shape_type == "rectangle":
        (x1, y1), (x2, y2) = points
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        new_point = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for p in new_point:
            tmp_point = (p[0] / width, p[1] / height)
            new_points.append(tmp_point)
    elif shape_type == "circle":
        bearing_angles = [
            0,
            15,
            30,
            45,
            60,
            75,
            90,
            105,
            120,
            135,
            150,
            165,
            180,
            195,
            210,
            225,
            240,
            255,
            270,
            285,
            300,
            315,
            330,
            345,
            360,
        ]

        orig_x1 = points[0][0]
        orig_y1 = points[0][1]

        orig_x2 = points[1][0]
        orig_y2 = points[1][1]

        radius = math.sqrt((orig_x2 - orig_x1) ** 2 + (orig_y2 - orig_y1) ** 2)

        circle_polygon = []

        for i in range(0, len(bearing_angles) - 1):
            ad1 = math.radians(bearing_angles[i])
            x1 = radius * math.cos(ad1)
            y1 = radius * math.sin(ad1)
            circle_polygon.append(((orig_x1 + x1) / height, (orig_y1 + y1) / width))

            ad2 = math.radians(bearing_angles[i + 1])
            x2 = radius * math.cos(ad2)
            y2 = radius * math.sin(ad2)
            circle_polygon.append(((orig_x1 + x2) / height, (orig_y1 + y2) / width))

        new_points = circle_polygon
    return new_points


def loadjson(source_folder_path, target_folder_path):
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)

    json_path_list = glob_extensions(source_folder_path, [".json", ".cson"])
    # json_demo = load_data(json_path_list[0])
    json_list = []
    for json_path in json_path_list:
        json_list.append(load_data(json_path))
    # json_list = [load_data(open(x)) for x in json_path_list]
    return json_list


def copyimage(source_folder_path, target_folder_path, ext):
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
    img_path_list = glob_extensions(source_folder_path, ext)
    for img_path in img_path_list:
        shutil.copy2(img_path, target_folder_path)


def labelme2yolo(result, category_map):
    # logger = get_root_logger()

    category_list = list(category_map.values())

    category_list = sorted(set(category_list), key=category_list.index)

    if "屏蔽" in category_list:
        category_list.remove("屏蔽")

    data_results = []
    num = 0
    with tqdm(result, desc="labelme2yolo") as pbar:
        for idx, data in enumerate(pbar):
            annotations = []
            img_filename = data["imagePath"]
            height, width = data["imageHeight"], data["imageWidth"]

            for shape in data["shapes"]:
                if shape["label"] not in category_map:
                    num += 1
                    # logger.warning(f'发现未知标签', shape['label'])
                    continue
                if category_map[shape["label"]] == "屏蔽":
                    continue

                new_points = []
                try:
                    new_points = shape2points(shape, height, width)
                except:
                    # logger.error(traceback.format_exc())
                    pass

                if len(new_points) == 0:
                    # logger.warning(
                    #     f'解析 shape 失败： {shape}，图片路径为{img_filename}')
                    continue

                category_id = category_list.index(category_map[shape["label"]])
                data_anno = dict(category_id=category_id, points=new_points)

                annotations.append(data_anno)

            # categories = [{'id': i, 'name': x} for i, x in enumerate(category_list)]
            data_dict = dict(images=img_filename, annotations=annotations)
            data_results.append(data_dict)

    return data_results


def save_txt(data_results, savepath):
    for data in data_results:
        file_name, _ext_list0 = os.path.splitext(data["images"])
        # file_name = file_name.replace('_G', '')
        for k, d in enumerate(data["annotations"]):
            with open(f"{savepath}/{file_name}.txt", "a") as f:
                id = d["category_id"]
                data0 = " ".join([f"{x} {y}" for x, y in d["points"]])
                line = str(id) + " " + data0
                f.write(line + "\n")


if __name__ == "__main__":
    work_dir = r"\\nas2.gzsonic.com\data\14-调试数据\szc\焊印"
    image_dir = r"\\nas2.gzsonic.com\data\14-调试数据\szc\焊印"
    label_dir = r"D:\pic\hanyin"

    category_map = {"焊印": "hanyin", "铜极焊印": "hanyin", "铝极焊印": "hanyin"}
    print("get path")
    results = loadjson(work_dir, image_dir)
    print("get path ok, convert")
    data_results = labelme2yolo(results, category_map)
    print("convert ok")
    save_txt(data_results, label_dir)
    # copyimage(work_dir, image_dir, ext='.bmp')
