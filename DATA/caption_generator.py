import os
import math
import json
import random
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from natsort import natsorted
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

total_count = 0
obj_id = 0
annotation_json_path = None
merge_data_dict = {
        "list_idx":[],
        "data_list":[],
        "alias":[]
    }

def load_json(json_path):
    with open(json_path,'r') as json_file:

        file = json.load(json_file)
        
    return file

def file_list(file_path, ext='.json'):
    f_list =[]
    for each_file in os.listdir(file_path):
        file_ext = os.path.splitext(each_file)[1]

        if file_ext in [ext]:
            f_list.append(os.path.join(file_path, each_file))
    
    f_list = natsorted(f_list)

    return f_list

def fit_box(img):
    h, w, _ = img.shape
    # vertical_zero_points = np.where((img == (0,0,0,0)).all(axis=2).all(axis=1))[0]
    # horizontal_zero_points = np.where((img == (0,0,0,0)).all(axis=2).T.all(axis=1) == True)[0]
    vertical_zero_points = np.where((img == (0,0,0)).all(axis=2).all(axis=1))[0]
    horizontal_zero_points = np.where((img == (0,0,0)).all(axis=2).T.all(axis=1) == True)[0]
    half_point = int(w/2)

    # 상단에서 0이 나오는 최고점
    y_min = np.max(vertical_zero_points[np.where(vertical_zero_points <= half_point)])
    y_max = np.min(vertical_zero_points[np.where(vertical_zero_points >= half_point)])
    x_min = np.max(horizontal_zero_points[np.where(horizontal_zero_points <= half_point)])
    x_max = np.min(horizontal_zero_points[np.where(horizontal_zero_points >= half_point)])
    
    return (x_min,y_min,x_max,y_max)

def extract_central_object(img):
    # 이미지 로드
    # img = cv2.imread(image_path)
    
    # 이미지 중앙값 계산
    h, w, _ = img.shape
    center = (w // 2, h // 2)
    
    # 검은색 배경 제외하고 객체만 있는 영역 찾기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 객체의 경계 박스 찾기 (중앙에 가까운 것으로 가정)
    max_area = 0
    best_rect = (0, 0, w, h)  # 기본값으로 전체 이미지 설정
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > max_area:
            max_area = area
            best_rect = (x, y, w, h)
    
    # 가장 큰 객체만 추출
    x, y, w, h = best_rect
    central_object = img[y:y+h, x:x+w]
    
    return best_rect

def rotate(origin, point, radian): # origin을 중심으로 point를 angle(radian) 한 값이 (qx,qy) 

    ox, oy = origin 
    px, py = point
    
    qx = ox + math.cos(radian) * (px - ox) - math.sin(radian) * (py - oy)
    qy = oy + math.sin(radian) * (px - ox) + math.cos(radian) * (py - oy)
    
    return qx, qy

def rotate_box(x_cen, y_cen, width, height, theta):

    x_min = x_cen-width/2
    y_min = y_cen-height/2
    rotated_x1,rotated_y1=rotate((x_cen,y_cen),(x_min,y_min),theta)
    rotated_x2,rotated_y2=rotate((x_cen,y_cen),(x_min,y_min+height),theta)
    rotated_x3,rotated_y3=rotate((x_cen,y_cen),(x_min+width,y_min+height),theta)
    rotated_x4,rotated_y4=rotate((x_cen,y_cen),(x_min+width,y_min),theta)

    return_arr = [[rotated_x1,rotated_y1],[rotated_x2,rotated_y2],[rotated_x3,rotated_y3],[rotated_x4,rotated_y4]]
    # print(return_arr)
    return return_arr

def make_square(img):

    # ZEROS = (0,0,0,0)
    ZEROS = (0,0,0)

    h, w, cc = img.shape

    triangle = int(math.sqrt(w**2 + h**2))+10

    zero_pad = np.full((triangle,triangle,cc),ZEROS, dtype=np.uint8)

    xx = (triangle - w) // 2
    yy = (triangle - h) // 2

    zero_pad[yy:yy+h,xx:xx+w,:] = img

    return zero_pad, (xx, yy)

def padding_cut(img,bbox):
    # img_copy = img.copy()
    # img_copy = img_copy[int(bbox[0][1]):int(bbox[0][3]),int(bbox[0][0]):int(bbox[0][2]),:]
    return img[int(bbox[0][1]):int(bbox[0][3]),int(bbox[0][0]):int(bbox[0][2]),:]

def file_random_list(file_path, ext='.json',percentage=10):
    f_list =[]
    print("file_path: ",file_path)
    for each_file in os.listdir(file_path):
        file_ext = os.path.splitext(each_file)[1]

        if file_ext in [ext]:
            f_list.append(os.path.join(file_path, each_file))
    
    f_list = natsorted(f_list)
    sample_size = max(1, len(f_list) * percentage // 100)
    sample_size = min(len(f_list), max(sample_size, 0))
    return random.sample(f_list, sample_size)

def file_random_split_list(file_path, ext='.json', percentage=10):
    f_list = []
    print("file_path: ", file_path)
    for each_file in os.listdir(file_path):
        file_ext = os.path.splitext(each_file)[1]
        if file_ext == ext:
            f_list.append(os.path.join(file_path, each_file))
    
    f_list = natsorted(f_list)
    sample_size = max(1, len(f_list) * percentage // 100)
    sample_size = min(len(f_list), max(sample_size, 0))
    
    # 10% 리스트 무작위 선택
    selected_samples = random.sample(f_list, sample_size)
    # 나머지 90% 리스트 생성
    remaining_samples = [item for item in f_list if item not in selected_samples]
    
    return selected_samples, remaining_samples

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

def set_categories():
    pig_color = ["brown","white","black"]
    pig_size = ["sow", "weaningPig","piglet","growingPig","finishingPig"]
    pig_pose = ["lateral_lying","standing","sternal_lying","sitting"]
    
    idx = 0
    categories = []
    for color in pig_color:
        for size in pig_size:
            for pose in pig_pose:
                category = {
                    "id": idx,
                    "name": f"{color}_{size}_{pose}",
                    "keypoints": [ "nose", "neck", "back1", "L_shoulder", "R_shoulder", "F_pit", "back2", "back3", "hip"],
                    "skeleton": [[1,2], [2,3], [3,4], [3,5], [3,6], [3,7], [7,8], [8,9]]
                }
                idx += 1
                categories.append(category)

    poses = [
        {
            "id": 0,
            "name": "lateral_lying"
        },
        {
            "id": 1,
            "name": "standing"
        },
        {
            "id": 2,
            "name": "sternal_lying"
        },
        {
            "id": 3,
            "name": "sitting"
        }
    ]
    return categories, poses     

def drawn_image(input_json, input_image, output_drawn_image):
    json_data = load_json(input_json)
    count = 0
    
    category_dic = {}
    for cate in json_data['categories']:
        category_dic[cate['id']] = cate["name"]
        
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

                test = cv2.putText(test, category_dic[da['category_id']], (int(cx - 10), int(cy - 10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, object_color, 1)
            else:
                test = image
        
        cv2.imwrite(os.path.join(output_drawn_image,"{0}".format(im['file_name'])), test)
        
def id_to_random_color(object_id):
    np.random.seed(object_id)
    return tuple([int(x) for x in np.random.randint(0, 255, 3)])

def General_WeightFunction(body_cm, shoulder_cm):

	return pow(body_cm*0.257, 1.5669) * pow(shoulder_cm*0.257, 0.9992) * 0.00139021064

    # corrp = 1 + (24 - 15) * 0.0098

    # return pow(body_cm*corrp, 1.7561) * pow(shoulder_cm*corrp, 0.7069) * 0.00152829888

# cm 구하는 방법
# body_dist  = (
# 		math.sqrt(pow((new_track.landmarks_x3 - new_track.landmarks_x2), 2) + pow((new_track.landmarks_y3 - new_track.landmarks_y2), 2))
# 		+ math.sqrt(pow((new_track.landmarks_x7 - new_track.landmarks_x3), 2) + pow((new_track.landmarks_y7 - new_track.landmarks_y3), 2))
# 		+ math.sqrt(pow((new_track.landmarks_x8 - new_track.landmarks_x7), 2) + pow((new_track.landmarks_y8 - new_track.landmarks_y7), 2))
# 		+ math.sqrt(pow((new_track.landmarks_x9 - new_track.landmarks_x8), 2) + pow((new_track.landmarks_y9 - new_track.landmarks_y8), 2))
# 	)
# kpt_shoulder = math.sqrt(pow((new_track.landmarks_x4 - new_track.landmarks_x5), 2) + pow((new_track.landmarks_y4 - new_track.landmarks_y5), 2))

def calculate_histogram(image):
    histogram = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    # try:
    #     image = cv2.resize(image, (64,64))
    #     histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # except:
    #     histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    return cv2.normalize(histogram, histogram).flatten()

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    # 두 히스토그램 간의 유사도를 계산
    return cv2.compareHist(hist1, hist2, method)

def category_mapping(pose,weight, new_color):
# def category_mapping(pose,weight, new_color, standard_color, kmeans, label_list, scaler):
    pig_size = None
    
    if weight >= 182:
        pig_size = "sow"
    elif weight < 13.6:
        pig_size = "weaningPig"
    elif weight >= 13.6 and weight < 34:
        pig_size = "piglet"
    elif weight >= 34 and weight < 68:
        pig_size = "growingPig"
    elif weight >= 68 and weight < 182:
        pig_size = "finishingPig"
    
    # hist_new = calculate_histogram(new_color)
    # hist_new_scal = scaler.transform([hist_new])
    # pred = kmeans.predict(hist_new_scal)
    # most_similarities_color = label_list[pred[0]]
    # similarities = {color: np.mean([compare_histograms(hist_new, calculate_histogram(img)) for img in images]) for color, images in standard_color.items()}
    # most_similarities_color = max(similarities, key=similarities.get)
    
    category_map = f"{new_color}_{pig_size}_{pose}"
    return category_map

def find_centermost_object(annotations, image):
   
    image_height, image_width, _ = image.shape

    # 각 객체의 중심에서 이미지 중심까지의 거리를 계산합니다.
    distances = []
    for annotation in annotations:
        cx, cy, _, _, _ = annotation['rbbox']
        distance = ((cx - image_width / 2) ** 2 + (cy - image_height / 2) ** 2) ** 0.5
        distances.append(distance)

    # 가장 짧은 거리를 가진 객체의 인덱스를 찾습니다.
    centermost_index = distances.index(min(distances))

    # 가장 중심에 가까운 객체를 반환합니다.
    return annotations[centermost_index]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
       
def combine_total_annotion():
    global total_count, obj_id, annotation_json_path, merge_data_dict
    
    # 기존 핸들러 제거 및 새 로그 파일 설정
    logger = logging.getLogger()  # root 로거를 가져옵니다.
    for handler in logger.handlers[:]:  # 기존 핸들러 목록을 복사하며 반복
        logger.removeHandler(handler)  # 각 핸들러를 제거
        handler.close()  # 핸들러 리소스를 명시적으로 해제
    LOG_FILE="combine.log"    
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    real_total_merge_data = []
    json_num=0
    exel_path = "./DATA/farm_alias_unknown.xlsx"
    # sample test 용도
    # exel_path = "./DATA/farm_alias_unknown_simple_test_2.xlsx"
    
    df = pd.read_excel(exel_path)
    
    data_dict ={
        "folder_path":df.iloc[:,3],
        "folder_alias":df.iloc[:,2]
    }
    
    fist_json_num = 0
    last_json_num = 0
    dict_idx = 0
    for data_path in data_dict["folder_path"]:
        print("json_path: ",os.path.join(data_path,'json'))
        
        logging.info("json_path: "+str(os.path.join(data_path,'json')))
        if os.path.exists(data_path):
            if "unknown" in data_path:
                if os.path.exists(os.path.join(data_path,'json')):
                    json_data_path=os.path.join(data_path,'json')
                    real_total_merge_data.extend(file_list(json_data_path, ext=".json"))
                    print(len(os.listdir(json_data_path)),"장")
                    logging.info(str(len(os.listdir(json_data_path)))+"장")
                    json_num=json_num+len(os.listdir(json_data_path))
                    if len(os.listdir(json_data_path)) > 0:
                        fist_json_num = last_json_num
                        last_json_num += len(os.listdir(json_data_path))
                        merge_data_dict['list_idx'].append([fist_json_num,last_json_num-1])
                        merge_data_dict['data_list'].append(file_list(json_data_path, ext=".json"))
                        merge_data_dict['alias'].append(data_dict["folder_alias"][dict_idx])
                
            for data_dir in os.listdir(data_path):
                if os.path.exists(os.path.join((os.path.join(data_path,data_dir)),'json')):
                    json_data_path=os.path.join((os.path.join(data_path,data_dir)),'json')
                    real_total_merge_data.extend(file_list(json_data_path, ext=".json"))
                    print(len(os.listdir(json_data_path)),"장")
                    logging.info(str(len(os.listdir(json_data_path)))+"장")
                    json_num=json_num+len(os.listdir(json_data_path))
                    if len(os.listdir(json_data_path)) > 0:
                        fist_json_num = last_json_num
                        last_json_num += len(os.listdir(json_data_path))
                        merge_data_dict['list_idx'].append([fist_json_num,last_json_num-1])
                        merge_data_dict['data_list'].append(file_list(json_data_path, ext=".json"))
                        merge_data_dict['alias'].append(data_dict["folder_alias"][dict_idx])
        else:
            print(f"Not exists {data_path}")
        dict_idx += 1
    
    annotation_json_path="/DL_data_super_ssd/new_EFID2023/9kpt_combine_new_hallway_"+str(json_num)+"_caption"
    if not os.path.exists(annotation_json_path):
        os.makedirs(annotation_json_path, exist_ok=True)
        os.makedirs(os.path.join(annotation_json_path,"image"), exist_ok=True)
        os.makedirs(os.path.join(annotation_json_path,"json"), exist_ok=True)
        os.makedirs(os.path.join(annotation_json_path,"drawn_image"), exist_ok=True)
    print(json_num)
    
    image_list = []
    anno_list = []
    
    # standard_list = file_list("./DATA/standard_crop_img",ext='.jpg')
    # standard_list = file_list("./DATA/standard_crop_img_center",ext='.jpg')
    # brown_list, white_list, black_list = [[cv2.imread(x) for x in standard_list if color in x] for color in ["brown", "white", "black"]]
    # standard_color = {"brown": brown_list, "white": white_list, "black": black_list}
    categories, poses = set_categories()
    color_list = []
    hist_list = []
    
    # for color, images in standard_color.items():
    #     for img in images:
    #         color_list.append(color)
    #         hist_list.append(calculate_histogram(img))

    model = Net()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()  # 평가 모드로 설정
    
    color_dic = {0:"black", 1:"brown", 2:"white"}
    
    # scaler = StandardScaler()
    # features = scaler.fit_transform(hist_list)
    
    # kmeans = KMeans(n_clusters=3, random_state=42)
    # kmeans.fit(features)
    
    total_count = 0
    obj_id = 0
    
    for each_json in tqdm(real_total_merge_data, total=len(real_total_merge_data) , desc="Processing", unit="it", unit_scale=True):
        
        pose_dict = {"lateral_lying":0,"standing":0,"sternal_lying":0,"sitting":0}
        try:
            new_json_data = load_json(each_json)
        except:
            print(f"json load faile = {each_json}")
            continue

        if len(new_json_data['annotations']) <= 0:
            continue

        source_file =new_json_data['images']['file_name']
        if os.path.exists(new_json_data['images']['file_name']):
            image = cv2.imread(new_json_data['images']['file_name'])
        else:
            print("없음 없음 없음 !!!::",new_json_data['images']['file_name'])
            logging.info("없음 없음 없음 !!!::"+new_json_data['images']['file_name'])
        

        obj_value = find_centermost_object(new_json_data['annotations'], image)
        
        body_dist  = (
                math.sqrt(pow((obj_value['keypoints'][6] - obj_value['keypoints'][3]), 2) + pow((obj_value['keypoints'][7] - obj_value['keypoints'][4]), 2))
                + math.sqrt(pow((obj_value['keypoints'][18] - obj_value['keypoints'][6]), 2) + pow((obj_value['keypoints'][19] - obj_value['keypoints'][7]), 2))
                + math.sqrt(pow((obj_value['keypoints'][21] - obj_value['keypoints'][18]), 2) + pow((obj_value['keypoints'][22] - obj_value['keypoints'][19]), 2))
                + math.sqrt(pow((obj_value['keypoints'][24]- obj_value['keypoints'][21]), 2) + pow((obj_value['keypoints'][25] - obj_value['keypoints'][22]), 2))
            )
        kpt_shoulder = math.sqrt(pow((obj_value['keypoints'][9] - obj_value['keypoints'][12]), 2) + pow((obj_value['keypoints'][10] - obj_value['keypoints'][13]), 2))
        
        obj_weight = General_WeightFunction(body_dist, kpt_shoulder)
        
        for value in new_json_data['annotations']:
            
            
            image_cp = image
            cx,cy,w,h,rad = value['rbbox']
            rotate_arr = rotate_box(cx,cy,w,h,rad)
            rotate_arr = np.array(rotate_arr,np.int32)
            mask = np.zeros((image_cp.shape[0], image_cp.shape[1]))
            cv2.fillConvexPoly(mask, rotate_arr, 1)
            mask = mask.astype(np.bool_)

            out = np.zeros_like(image_cp)
            out[mask] = image_cp[mask]
            # cv2.imwrite("test_mask.jpg",out)
            x_min = rotate_arr[:,0].min()
            y_min = rotate_arr[:,1].min()
            x_max = rotate_arr[:,0].max()
            y_max = rotate_arr[:,1].max()
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max > image_cp.shape[1]:
                x_max = image_cp.shape[1]
            if y_max > image_cp.shape[0]:
                y_max = image_cp.shape[0]
            
            mask_img = out[y_min:y_max,x_min:x_max]
            degrees = math.degrees(rad)
            mask_img_sq = make_square(mask_img)
            mask_h, mask_w, _ = mask_img_sq[0].shape
            matrix = cv2.getRotationMatrix2D((mask_w/2,mask_h/2),degrees,1)
            mask_sq = cv2.warpAffine(mask_img_sq[0],matrix,(mask_w,mask_h))
            x,y,w,h = extract_central_object(mask_sq)
            target_img = mask_sq[y:y+h, x:x+w]
            # x1, y1, x2, y2 = fit_box(mask_sq)
            # bbox = [[x1,y1,x2,y2]]
            # target_img = padding_cut(mask_sq,bbox)
            
            f_h, f_w, _ = target_img.shape
            f_h_c = f_h//2
            f_w_c = f_w//2
            
            f_h_c_r = int(f_h_c * 0.17)
            f_w_c_r = int(f_w_c * 0.17)
            if f_h > f_w:
                center_crop = target_img[f_h_c-f_h_c_r:f_h_c+f_h_c_r,f_w_c-f_h_c_r:f_w_c+f_h_c_r]
            else:
                center_crop = target_img[f_h_c-f_w_c_r:f_h_c+f_w_c_r,f_w_c-f_w_c_r:f_w_c+f_w_c_r]
            try:
                target_img = cv2.resize(target_img,(224,224))
            except:
                print("a")
            target_img = torch.tensor(target_img,dtype=torch.float)
            target_img = target_img.permute(2,0,1)
            target_img = target_img.unsqueeze(0)
            output = model(target_img)
            # center_crop = torch.tensor(center_crop,dtype=torch.float)
            # center_crop = center_crop.permute(2,0,1)
            # center_crop = center_crop.unsqueeze(0)
            # output = model(center_crop)
            values, pred = torch.max(output,1)
            obj_color = color_dic[pred.item()]
            
            # cm 구하는 방법
            # body_dist  = (
            #         math.sqrt(pow((value['keypoints'][6] - value['keypoints'][3]), 2) + pow((value['keypoints'][7] - value['keypoints'][4]), 2))
            #         + math.sqrt(pow((value['keypoints'][18] - value['keypoints'][6]), 2) + pow((value['keypoints'][19] - value['keypoints'][7]), 2))
            #         + math.sqrt(pow((value['keypoints'][21] - value['keypoints'][18]), 2) + pow((value['keypoints'][22] - value['keypoints'][19]), 2))
            #         + math.sqrt(pow((value['keypoints'][24]- value['keypoints'][21]), 2) + pow((value['keypoints'][25] - value['keypoints'][22]), 2))
            #     )
            # kpt_shoulder = math.sqrt(pow((value['keypoints'][9] - value['keypoints'][12]), 2) + pow((value['keypoints'][10] - value['keypoints'][13]), 2))
            
            # obj_weight = General_WeightFunction(body_dist, kpt_shoulder)
            
            obj_pose = poses[value['pose_id']]['name']
            pose_dict[obj_pose] += 1
            
            category_info = category_mapping(obj_pose,obj_weight, obj_color)
            # category_info = category_mapping(obj_pose,obj_weight, target_img, standard_color, kmeans, color_list, scaler)
            # category_info = category_mapping(obj_pose,obj_weight, center_crop, standard_color, kmeans, color_list, scaler)
            
            category_id = [index for index, item in enumerate(categories) if item["name"] == category_info]
            anno_json = {
                'id': obj_id,
                'image_id': total_count,
                'category_id': category_id[0],
                'pose_id': value['pose_id'],
                'num_keypoints': 9,
                'bbox': value['bbox'],
                'rbbox': value['rbbox'],
                'keypoints': value['keypoints'],
                'segmentation': value['segmentation'],
                'segmentation_rbbox': value['segmentation_rbbox'],
                'area': value['area'],
                'iscrowd': value['iscrowd']
            }
            
            anno_list.append(anno_json)
            
            obj_id += 1
 
        img_alias = None
        for idx,list_idx in enumerate(merge_data_dict['list_idx']):
            if list_idx[0] <= total_count <= list_idx[1] and img_alias==None:
                img_alias = merge_data_dict["alias"][idx]
                break
 
        filtered_list = [(k, v) for k, v in pose_dict.items() if v > 0]

        # Building the script based on the filtered_list content
        script_parts = []
        tot_obj = 0
        for pose, count in filtered_list:
            pose_name = pose.replace('_', ' ')
            if count > 1:
                script_parts.append(f"{count} pigs are in the {pose_name} pose")
            else:
                script_parts.append(f"{count} pig are in the {pose_name} pose")
            tot_obj += count

        if len(script_parts) > 1:
            script = ", ".join(script_parts[:-1]) + ", and " + script_parts[-1]
        elif len(script_parts) == 1:
            script = script_parts[0]
        else:
            script = "No poses with more than 0 pig found."
            
        script = f"In a hallway of farm {img_alias}, there are a total of {tot_obj} pigs, " + script + ", in a top view image."
 
        images_info = {
            "id": total_count,           
            "file_name": f'{str(total_count+1).zfill(6)}.jpg',
            "height": image.shape[0],
            "width": image.shape[1],
            "caption": f"{script}"
        }
        
        image_list.append(images_info)
 
        cv2.imwrite(os.path.join(os.path.join(annotation_json_path,"image"), '{0}.jpg'.format(str(total_count+1).zfill(6))), image)
        
        total_count += 1
    
    print(total_count)
    train_json = {}
    train_json['images'] = image_list
    train_json['annotations'] = anno_list
    train_json['categories'] = categories
    train_json['poses'] = poses
    with open(os.path.join(os.path.join(annotation_json_path,"json"),"annotation_caption.json"), 'w') as file:
        json.dump(train_json, file, indent=4)

    # annotation_json_path="/DL_data_super_ssd/new_EFID2023/9pkt_combine_new_hallway_2513_caption"
    drawn_image(os.path.join(os.path.join(annotation_json_path,"json"),"annotation_caption.json"), os.path.join(annotation_json_path,"image"), os.path.join(annotation_json_path,"drawn_image"))
    if os.path.exists('combine.log'):
        shutil.move('combine.log', annotation_json_path)
        
        
if __name__=="__main__":
    combine_total_annotion()
    