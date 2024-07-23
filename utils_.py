import hashlib
import json
import os
import re
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import find_objects
from skimage import measure
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
dcm_formats = ['dcm']  # acceptable dicom suffixes

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, np.ndarray):
            return list(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

class Utils(object):
    def __init__(self):
        super(Utils, self).__init__()

    @staticmethod
    def create_dirs(path, rmtree=False):
        if rmtree and os.path.exists(path): shutil.rmtree(path)
        try:
            if not os.path.exists(path): os.makedirs(path)
        except:
            pass

    def make_path(path, *subPath, rmtree=False):
        dst_path = os.path.join(path, "/".join(subPath))
        Utils().create_dirs(dst_path, rmtree)
        return dst_path
    
    @staticmethod
    def parseYoloTxt(txt_path):
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as fp:
                lines = np.array([x.split() for x in fp.read().splitlines()], dtype=np.float32)
            return lines
        else:
            return []
    
    @staticmethod
    def parseTxt(txt_path):
        with open(txt_path, "r", encoding="utf-8") as fp:
            lines = [x.strip() for x in fp.read().splitlines()]
        return lines
    
    @staticmethod
    def write2txt(txt_path, img_list):
        with open(txt_path, "w", encoding="utf-8") as f:
            for img in img_list:
                f.write("{}\n".format(img))

    @staticmethod
    def openJsonFile(json_path):
        with open(json_path, "r", encoding="utf-8") as jf:
            json_data = json.load(jf)
        return json_data
    
    @staticmethod
    def saveJsonFile(json_path, json_data, encoding="utf-8", indent=2):
        with open(json_path, "w", encoding=encoding) as jf:
            json.dump(json_data, jf, ensure_ascii=False, indent=indent)

    @staticmethod
    def saveExcel(save_path, data_info, sheet_name="test", col_name=[], index=False, mode="w"):
        """保存数据到xlsx

        Args:
            save_path (_type_): 保存的xlsx路径
            data_info (_type_): 待保存的数组
            sheet_name (str, optional): 保存的sheet名称. Defaults to "test".
            col_name (list, optional): 指定的列名称. Defaults to [].
            index (bool, optional): 是否保存左侧数量列. Defaults to False.
            mode (str, optional): 写方式w/a. Defaults to "w".
        """
        with pd.ExcelWriter(save_path, engine='openpyxl', mode=mode) as xlsx:
            dt = pd.DataFrame(data_info, columns=col_name)
            dt.to_excel(xlsx, index=index, sheet_name=sheet_name)
    
    @staticmethod
    def readExcel(xlsx_path, sheet_name="Sheet1", header=None) -> dict:
        """读取指定sheet名称的xlsx

        Args:
            xlsx_path (_type_): _description_
            sheet_name (str, optional): _description_. Defaults to "test".

        Returns:
            _type_: numpy数组
        """
        if sheet_name != None:
            pd_data = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header).values
        else:
            pd_data = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header)
        return pd_data

     
    @staticmethod
    def check_file(file_path, method="md5"):
        if method == "md5":
            # 建立 MD5 物件
            md5 = hashlib.md5()
        elif method == "sha1":
            # 建立 SHA1 物件
            md5 = hashlib.sha1()
        elif method == "sha256":
            # 建立 SHA256 物件
            md5 = hashlib.sha256()

        # 计算文件的 MD5 检查码
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)

        # 取得 MD5 结果
        digest = md5.hexdigest()
        return digest
    
    @staticmethod
    def generate_md5(string, method="md5"):
        if method == "md5":
            # 建立 MD5 物件
            md5 = hashlib.md5()
        elif method == "sha1":
            # 建立 SHA1 物件
            md5 = hashlib.sha1()
        elif method == "sha256":
            # 建立 SHA256 物件
            md5 = hashlib.sha256()
        else:
            md5 = hashlib.sha224()
            
        md5.update(string.encode('utf-8'))
        md5_value = md5.hexdigest()
        return md5_value
    
    @staticmethod
    def LetterBoxResize(ori_image, imgsz=(416, 416), color=(128, 128, 128)):
        INPUT_W = imgsz[0]
        INPUT_H = imgsz[1]

        h, w, c = ori_image.shape
        # image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = ori_image
        # Calculate widht and height and paddings
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, value=color
        )

        r = min(r_w, r_h)
        ratio = r, r  # width, height ratios
        dw, dh = INPUT_W - tw, INPUT_H - th  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        return image, ratio, (dw, dh)

    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    @staticmethod
    def crop_pure_us_img(img_ori):
        # input: img, mask
        # output: img_crop, mask_crop, crop_box:[xmin, xmax, ymin, ymax]

        img = img_ori.copy()

        img_top_region = 100
        img_bottom_region = 50
        img[:img_top_region, :] = 0
        img[-img_bottom_region:, :] = 0

        threshold = 2
        img[img < threshold] = 0
        img[img >= threshold] = 1

        tImg = img
        tImg_map = measure.label(tImg.astype(dtype=np.int), connectivity=2)
        tImg_stats = measure.regionprops(tImg_map)
        tImg_len = len(tImg_stats)
        tImg_area = np.zeros((tImg_len))
        result = np.zeros((img.shape))

        for tImg_num in range(tImg_len):
            tImg_area[tImg_num] = tImg_stats[tImg_num].area

        index_1st_max = np.argmax(tImg_area)
        result[tImg_map == (index_1st_max + 1)] = 1

        max_us_region_ratio = tImg_area[index_1st_max] / np.sum(tImg_area)

        if max_us_region_ratio < 0.8:
            tImg_area[index_1st_max] = 0
            index_2nd_max = np.argmax(tImg_area)
            result[tImg_map == (index_2nd_max + 1)] = 1

        loc = find_objects(result, max_label=1)

        ymin = int(loc[0][0].start)
        if ymin == img_top_region:
            ymin = np.int(img_top_region / 2)
        ymax = int(loc[0][0].stop)

        xmin = int(loc[0][1].start)
        xmax = int(loc[0][1].stop)

        img_crop = img_ori[ymin:ymax, xmin:xmax]
        # crop_box = [xmin, xmax, ymin, ymax]
        crop_box = [xmin, ymin, xmax - xmin, ymax - ymin]
        return img_crop, crop_box

    @staticmethod
    def coco_xywh2yolo_xywh(size, box):
        """将ROI的坐标转换为yolo需要的坐标
        :param size: size是图片的w和h
        :param box: box里保存的是ROI的坐标(x, y)的最大值和最小值
        :return: 返回值为ROI中心点相对于图片大小的比例坐标，和ROI的w, h相对于图片大小的比例
        """
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = box[0] + box[2] / 2.0
        y = box[1] + box[3] / 2.0
        w = box[2]
        h = box[3]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    @staticmethod
    def yolo_xywh2coco_xywh(size, box, xywh=True):
        rw, rh = size
        
        # 第一步：还原box的中心点、宽和搞
        cx, cy = box[0] * rw, box[1] * rh
        bw, bh = box[2] * rw, box[3] * rh
        # 第二步：计算左上角点和右下角点
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = x1 + bw, y1 + bh
        
        if xywh:
            return x1, y1, bw, bh        
        else:
            return x1, y1, x2, y2
    
    @staticmethod
    def bbox_padding(box, ratio, pad, xywh=True):
        x1, y1, x2, y2 = box
        dw, dh = pad
        
        bx1 = float(ratio[0] * x1 + dw)  # pad width
        by1 = float(ratio[1] * y1 + dh)  # pad height
        bx2 = float(ratio[0] * x2 + dw)
        by2 = float(ratio[1] * y2 + dh)
        bw, bh = bx2 - bx1, by2 - by1
        
        if xywh:
            return bx1, by1, bw, bh
        else:
            return bx1, by1, bx2, by2
    
    @staticmethod
    def bbox_padding_revert(box, ratio, pad, xywh=True):
        x1, y1, x2, y2 = box
        dw, dh = pad
        
        bx1 = (x1 - dw) / ratio[0]
        by1 = (y1 - dh) / ratio[1]
        bx2 = (x2 - dw) / ratio[0]
        by2 = (y2 - dh) / ratio[1]
        
        # bx1 = float(ratio[0] * x1 + dw)  # pad width
        # by1 = float(ratio[1] * y1 + dh)  # pad height
        # bx2 = float(ratio[0] * x2 + dw)
        # by2 = float(ratio[1] * y2 + dh)
        bw, bh = bx2 - bx1, by2 - by1
        
        if xywh:
            return bx1, by1, bw, bh
        else:
            return bx1, by1, bx2, by2
    
    @staticmethod
    def calc_iou(bbox1, bbox2):
        if not isinstance(bbox1, np.ndarray):
            bbox1 = np.array(bbox1)
        if not isinstance(bbox2, np.ndarray):
            bbox2 = np.array(bbox2)
        xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
        xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

        ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
        xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
        ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
        xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

        h = np.maximum(ymax - ymin, 0)
        w = np.maximum(xmax - xmin, 0)
        intersect = h * w

        union = area1 + np.squeeze(area2, axis=-1) - intersect
        return intersect / union

    @staticmethod
    def bbox_convert(bbox):
        """
        box 坐标 float 转 Int
        """
        return np.array(list(map(int, bbox)))

    @staticmethod
    def find_images(root_path, full_path=True):
        images = []
        for root, dirs, files in os.walk(root_path):
            for f in files:
                if full_path:
                    images.append(os.path.join(root, f))
                else:
                    images.append(f)

        return images

    @staticmethod
    def save_video(save_path, video_name, cap=None, fps=None, width=None, height=None):
        if cap is not None and fps is None: fps = int(cap.get(cv2.CAP_PROP_FPS))  # 返回视频的fps--帧率
        if cap is not None and width is None: width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 返回视频的宽
        if cap is not None and height is None: height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 返回视频的高

        fourcc = 'mp4v'  # output video codec
        vid_writer = cv2.VideoWriter(os.path.join(save_path, video_name + ".avi"),
                                     cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))

        return vid_writer

    @staticmethod
    def pad_img_border(image, scale=16 / 9):
        h, w, c = image.shape

        # 比例大于 16:9 跳过不作处理
        if w / h >= scale: return image, 0

        w_pad = int((h * scale - w) / 2)

        zero_array = np.zeros((h, w + w_pad * 2, c), dtype=np.uint8)
        zero_array[:, w_pad: w_pad + w, :] = image

        return zero_array, w_pad
    
    @staticmethod
    def box_iou(boxA, boxB, convert=True):
        if convert:
            boxA = [int(x) for x in boxA]
            boxB = [int(x) for x in boxB]
        else:
            boxA = [x for x in boxA]
            boxB = [x for x in boxB]
        
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou, interArea

    @staticmethod
    def box_inter_area(boxA, boxB):
        boxA = [int(x) for x in boxA]
        boxB = [int(x) for x in boxB]

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxBArea)

    @staticmethod
    def mask_inter_area(maskA, maskB):
        if isinstance(maskB, (tuple, list)):
            x1, y1, x2, y2 = map(int, maskB)
            maskB = np.zeros_like(maskA, dtype=np.uint8)
            maskB[y1:y2, x1:x2] = 1
        
        smooth = 1e-9
        interArea = (maskA * maskB).sum()
        return interArea / (maskB.sum() + smooth)

    @staticmethod
    def natural_key(string_):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]
    
    @staticmethod
    def change_owner(path):
        # id guofeng 查看
        # uid=999(guofeng) gid=998(guofeng) groups=998(guofeng),999(docker)
        # uid=1003(guofeng) gid=1003(guofeng) groups=1003(guofeng),27(sudo),999(docker)
        for root, dirs, files in os.walk(path):
            os.chown(root, 1003, 1003)
            for f in files:
                # path uid gid
                os.chown(os.path.join(root, f), 1003, 1003)
    
    @staticmethod
    def find_max_contour(gray_img):
        # if img.shape[-1] == 3:
        #     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray_img = img
        cnts, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        index = np.argmax([cv2.contourArea(c) for c in cnts])

        return cnts[index]
    
    @staticmethod
    def train_test_val_split(img_paths, ratio_train=0.8, ratio_test=0.1, ratio_val=0.1, seed=42):
        assert int(ratio_train + ratio_test + ratio_val) == 1
        train_img, middle_img = train_test_split(img_paths, test_size=1-ratio_train, random_state=0)
        ratio = ratio_val / (1 - ratio_train)
        val_img, test_img = train_test_split(middle_img, test_size=ratio, random_state=seed)
        print("NUMS of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
        return train_img, val_img, test_img
    
    @staticmethod
    def train_val_split(img_paths, ratio_train=0.8, seed=42):
        train_img, val_img = train_test_split(img_paths, test_size=1-ratio_train, random_state=seed)
        print("NUMS of train:val = {}:{}".format(len(train_img), len(val_img)))
        return train_img, val_img
    
    @staticmethod
    def gen_coco_anno(coco_anno_path, imgs_info, setname="train", classes=[]):
        """根据图像数据字典生成CoCo格式JSON

        Args:
            coco_anno_path (_type_): _description_
            imgs_info (_type_): _description_
            setname (str, optional): _description_. Defaults to "train".
            classes (list, optional): _description_. Defaults to [].
        """           
        dataset = {'categories': [], 'images': [], 'annotations': []}
        for i, cls in enumerate(classes):
            dataset['categories'].append({'id': i, 'name': cls})

        ann_id_cnt = 0
        img_id_cnt = 0
        for img_name, img_info in imgs_info.items():
            height, width, _ = img_info["shape"]

            dataset['images'].append({'id': img_id_cnt,
                                      'file_name': img_name,
                                      'width': width,
                                      'height': height})

            for bi, bbox in enumerate(img_info["bboxes"]):
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                cls_id = int(classes.index(img_info["labels"][bi]))
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                dataset['annotations'].append({
                    'id': ann_id_cnt,
                    'image_id': img_id_cnt,
                    'category_id': cls_id,
                    'iscrowd': 0,
                    'area': w * h,
                    'bbox': [x1, y1, w, h],
                    # 'segmentation': img_info["rles"][bi]
                })
                ann_id_cnt += 1
            img_id_cnt += 1
            
        json_name = os.path.join(coco_anno_path, '{}.json'.format(setname))
        with open(json_name, 'w') as f:
            json.dump(dataset, f, indent=2, cls=MyEncoder)
            print('Save annotation to {}'.format(json_name))
    
    @staticmethod
    def save_img2id(root_dir, dataset):
        coco_anno_path = os.path.join(root_dir, "annotations")

        with open('%s/%s.json' % (coco_anno_path, dataset), "r") as file:
            coco_data = json.load(file)

        with open(r"%s/images2ids_%s.json" % (coco_anno_path, dataset), "w") as wf:
            images_dict = {}
            for item in coco_data["images"]:
                images_dict[item["file_name"]] = item["id"]

            json.dump(images_dict, wf, indent=2)
            print('Save images2id_{}.json'.format(dataset))
    
    @staticmethod
    def box_jitter(image, labels, jitter=(0.1, 0.1)):
        nl = np.zeros_like(labels)
        img_h, img_w = image.shape[:-1]
        for index, label in enumerate(labels):
            # x1, y1, x2, y2 = label
            # w, h = x2 - x1, y2 - y1
            x1, y1, w, h = label
            
            wf, hf = w * jitter[1] // 2, h * jitter[0] // 2
            x1 = x1 - wf
            y1 = y1 - hf
            x2 = x1 + w * (1 + jitter[1])
            y2 = y1 + h * (1 + jitter[0])
            # xc, yc = (x1 + x2) / 2, (y1 + y2) / 2

            # t = (yc - y1) * (1 - jitter)
            # l = (xc - x1) * (1 - jitter)
            # b = (y2 - yc) * (1 - jitter)
            # r = (x2 - xc) * (1 - jitter)

            # x1 = xc - l
            # y1 = yc - t
            # x2 = xc + r
            # y2 = yc + b

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            if y2 > y1 and x2 > x1:
                bbox = [x1, y1, x2, y2]
                nl[index] = [*bbox]
            else:
                nl[index] = labels[index]

        return nl
    
    @staticmethod
    def draw_rectangle(coordinates, img_data, save_path):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for coordinate in coordinates:
            left, top, right, bottom, label = map(int, coordinate)
            color = colors[label % len(colors)]
            cv2.rectangle(img_data, (left, top), (right, bottom), color, 2)
            cv2.putText(img_data, str(label), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        cv2.imwrite(save_path, img_data)
        
    @staticmethod
    def draw_rectanglev2(coordinates, img_data, thinkness=2, save_path=None):
        # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        colors = [(0, 255, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),]
        for coordinate in coordinates:
            left, top, right, bottom = map(int, coordinate[:4])
            label = int(coordinate[-1])
            color = colors[label % len(colors)]
            cv2.rectangle(img_data, (left, top), (right, bottom), color, thinkness)
            cv2.putText(img_data, "{}-{}".format(label, coordinate[-2]), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 
                        thinkness*0.6, color, thinkness)

        if save_path is not None: cv2.imwrite(save_path, img_data)
    
    @staticmethod
    def getSid2NameDict(csv_path : str, encoding="gbk"):
        # 获取序列号和图像名称的对应
        sid2name, name2sid, pat_imgs = {}, {}, {}
        if csv_path.endswith("csv"):
            pd_data = pd.read_csv(csv_path, encoding=encoding, header=None).values
        elif csv_path.endswith("xlsx"):
            pd_data = pd.read_excel(csv_path, header=None).values
        header = list(pd_data[0])
        csv_data = pd_data[1:, :]
        for line in tqdm(csv_data):
            sid = line[header.index("seriesinstanceUID")]
            img_name = line[header.index("originPath")].split("/")[-1]
            pat_name = line[header.index("originPath")].split("/")[0]
            sid2name[sid] = img_name
            name2sid[img_name] = sid
            if pat_name not in pat_imgs.keys(): pat_imgs[pat_name] = []
            pat_imgs[pat_name].append(img_name)
        return sid2name, name2sid, pat_imgs
    
    @staticmethod
    def getInterUnionDiff(list1, list2, name="inter"):
        if name == "inter":
            new_list = list(set(list1).intersection(set(list2)))
        elif name == "union":
            new_list = list(set(list1).union(set(list2)))
        elif name == "diff":
            new_list = list(set(list1).difference(set(list2)))
        else:
            print("name error={}".format(name))
        return new_list
        
    @staticmethod
    def multiThreadProcess(files, theading_num, function):
        with ThreadPoolExecutor(max_workers=theading_num) as executor:
            futures = [executor.submit(function, [file_path]) for file_path in files]
            results = [future.result() for future in tqdm(futures)]
        
        return results