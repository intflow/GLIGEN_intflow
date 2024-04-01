import json
import os
from natsort import natsorted
import cv2
from tqdm import tqdm
import random
import numpy as np
import argparse

def load_json(json_path):
    with open(json_path,'r') as json_file:
        file = json.load(json_file)
    return file

def get_xy_point(rbbox):
    cx, cy, width, height, radian = rbbox
    xmin, ymin = cx - (width - 1) / 2, cy - (height - 1) / 2
    xy1 = xmin, ymin
    xy2 = xmin, ymin + height - 1
    xy3 = xmin + width - 1, ymin + height - 1
    xy4 = xmin + width - 1, ymin
    cents = np.array([cx, cy])
    corners = np.stack([xy1, xy2, xy3, xy4])
    u = np.stack([np.cos(radian), -np.sin(radian)])
    l = np.stack([np.sin(radian), np.cos(radian)])
    R = np.vstack([u, l])
    corners = np.matmul(R, (corners - cents).transpose(1, 0)).transpose(1, 0) + cents
    return corners

def id_to_random_color(object_id):
    np.random.seed(object_id)
    return tuple([int(x) for x in np.random.randint(0, 255, 3)])

def drawn_image(input_json, input_image, output_drawn_image):
    json_data = load_json(input_json)
    count = 0
    for im in tqdm(json_data['images'], total=len(json_data['images'])):
        image = cv2.imread(os.path.join(input_image,'{0}'.format(im['file_name'])))
        for da in json_data['annotations']:
            if im['id'] == da['image_id']:
                object_color = id_to_random_color(da['id'])
                cx, cy, width, height, radian = da['rbbox']
                segmen = get_xy_point([cx, cy, width, height, radian])
                segmen = [np.array(segmen).flatten().tolist()]
                segmentation_rbbox = np.array(segmen).astype(int).reshape(4,2)
                test = cv2.drawContours(image,[segmentation_rbbox],0,color=object_color, thickness=3)
                keypoints = da['keypoints']
                for kpt_idx in range(0, len(keypoints), 3):  # keypoints는 (x, y, v) 세트를 가정합니다.
                    kpt_x, kpt_y = int(keypoints[kpt_idx]), int(keypoints[kpt_idx + 1])
                    test = cv2.circle(test, (kpt_x, kpt_y), radius=2, color=object_color, thickness=5)  # 개체 색상 사용
                    test = cv2.putText(test, str(kpt_idx // 3), (kpt_x, kpt_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, object_color, 3)
                if da['category_id'] == 0:
                    for kpts_set in json_data['categories'][0]['skeleton']:
                        kpt_idx1 = (int(keypoints[(kpts_set[0]-1) * 3]), int(keypoints[(kpts_set[0]-1) * 3+1]))
                        kpt_idx2 = (int(keypoints[(kpts_set[1]-1) * 3]), int(keypoints[(kpts_set[1]-1) * 3+1]))
                        test = cv2.line(test, kpt_idx1, kpt_idx2, [255, 255, 255], 1)
                elif da['category_id'] == 1:
                    for kpts_set in json_data['categories'][1]['skeleton']:
                        kpt_idx1 = (int(keypoints[(kpts_set[0]-1) * 3]), int(keypoints[(kpts_set[0]-1) * 3+1]))
                        kpt_idx2 = (int(keypoints[(kpts_set[1]-1) * 3]), int(keypoints[(kpts_set[1]-1) * 3+1]))
                        test = cv2.line(test, kpt_idx1, kpt_idx2, [255, 255, 255], 1)
                category_dic = {0:'lateral_lying', 1:'standing', 2:'sternal_lying', 3:'sitting'}
                test = cv2.putText(test, category_dic[da['category_id']], (int(cx - 10), int(cy - 10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, object_color, 1)
            else:
                test = image
        cv2.imwrite(os.path.join(output_drawn_image,"{0}".format(im['file_name'])), test)

def main():
    drawn_image(args.drawn_input_json, args.drawn_input_image, args.drawn_output_image)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("--input_json", type=str, default='/DL_data_super_ssd/new_EFID2023/count/raw_data/cj_vina/base_dataset/json')
    ap.add_argument("--drawn_input_json", type=str, default='/data/efc20k/json/annotation.json')
    ap.add_argument("--drawn_input_image", type=str, default='/data/efc20k/image')
    ap.add_argument("--drawn_output_image", type=str, default='/data/efc20k/drawn_image')
    args = ap.parse_args()
    main()