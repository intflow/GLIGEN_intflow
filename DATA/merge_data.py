import json
import os
from natsort import natsorted
import cv2
from tqdm import tqdm
import random
import numpy as np
import argparse
import logging
import shutil

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
    categories = [
        {
            "id": 0,
            "name": "pig",
            "keypoints": [ "nose", "neck", "back1", "L_shoulder", "R_shoulder", "F_pit", "back2", "back3", "hip"],
            "skeleton": [[1,2], [2,3], [3,4], [3,5], [3,6], [3,7], [7,8], [8,9]]
        }
    ]

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

def change_anno(input_file_path):
    # Replace 'coco_annotations.json' with your COCO JSON file path
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    # Assuming that the JSON file has a key 'annotations'
    if 'annotations' in data.keys():
        for annotation in data['annotations']:
            if 'category_id' in annotation.keys() and 'pose_id' in annotation.keys():
                annotation['category_id'] = annotation['pose_id']

    if 'categories' in data.keys() and 'poses' in data.keys():
        cat_pig = data['categories'][0]
        data['categories'] = []
        for pose in data['poses']:
            cat_pig_new = cat_pig.copy()
            cat_pig_new['id'] = pose['id']
            cat_pig_new['name'] = cat_pig_new['name'] + '_' + pose['name']
            data['categories'].append(cat_pig_new)

    # with open(os.path.join(os.path.dirname(input_file_path), 'annotation.json'), 'w') as f:
    #     json.dump(data, f, indent=4)
    with open(input_file_path, 'w') as f:
        json.dump(data, f, indent=4)      
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
        
def id_to_random_color(object_id):
    np.random.seed(object_id)
    return tuple([int(x) for x in np.random.randint(0, 255, 3)])
# def fod_combine_annotion(farm_id,farm_name):
#     with open(f"/home/intflow/works/auto_train/count_cam_id_info.json", 'r') as json_file:
#         data = json.load(json_file)
#     data_path=data[farm_id][1]
    
#     merge_data = []
#     if os.path.exists(os.path.join((os.path.join(data_path,'FOD')),'json')):
#         json_data_path=os.path.join((os.path.join(data_path,'FOD')),'json')
#         merge_data.extend(file_list(json_data_path, ext=".json"))    
#     annotation_json_path=f'/DL_data_super_ssd/new_EFID2023/count/training_data/FOD/{farm_name}'
#     os.makedirs(os.path.join(annotation_json_path,"image"), exist_ok=True)
#     os.makedirs(os.path.join(annotation_json_path,"json"), exist_ok=True)
#     os.makedirs(os.path.join(annotation_json_path,"drawn_image"), exist_ok=True)
#     image_list = []
#     anno_list = []
    
#     total_count = 0
#     obj_id = 0       
#     for each_json in tqdm(merge_data, total=len(merge_data)):
#         new_json_data = load_json(each_json)
        
#         # if "cj_vina_unknown" in each_json:
#         #     image = cv2.imread(each_json.replace("/single_json/", "/image/").replace(".json", ".jpg"))
#         # else:
#         source_file =new_json_data['images']['file_name']
#         if os.path.exists(new_json_data['images']['file_name']):
#             image = cv2.imread(new_json_data['images']['file_name'])
#         else:
#             print("없음 없음 없음 !!!::",new_json_data['images']['file_name'])
#         images_info = {
#             "id": total_count,
#             "file_name": '{0}.jpg'.format(str(total_count+1).zfill(6)),
#             "height": image.shape[0],
#             "width": image.shape[1]
#         }
        
#         image_list.append(images_info)
        
#         for value in new_json_data['annotations']:
#             anno_json = {
#                 'id': obj_id,
#                 'image_id': total_count,
#                 'category_id': value['category_id'],
#                 'pose_id': value['pose_id'],
#                 'num_keypoints': 9,
#                 'bbox': value['bbox'],
#                 'rbbox': value['rbbox'],
#                 'keypoints': value['keypoints'],
#                 'segmentation': value['segmentation'],
#                 'segmentation_rbbox': value['segmentation_rbbox'],
#                 'area': value['area'],
#                 'iscrowd': value['iscrowd']
#             }
            
#             anno_list.append(anno_json)
            
#             obj_id += 1
 
#         cv2.imwrite(os.path.join(os.path.join(annotation_json_path,"image"), '{0}.jpg'.format(str(total_count+1).zfill(6))), image)
        
#         total_count += 1
    
#     categories, poses = set_categories()
#     print(total_count)
#     train_json = {}
#     train_json['images'] = image_list
#     train_json['annotations'] = anno_list
#     train_json['categories'] = categories
#     train_json['poses'] = poses
#     with open(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), 'w') as file:
#         json.dump(train_json, file, indent=4)
        
#     change_anno(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"))
#     drawn_image(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), os.path.join(annotation_json_path,"image"), os.path.join(annotation_json_path,"drawn_image"))   

def combine_total_annotion():
    # 기존 핸들러 제거 및 새 로그 파일 설정
    logger = logging.getLogger()  # root 로거를 가져옵니다.
    for handler in logger.handlers[:]:  # 기존 핸들러 목록을 복사하며 반복
        logger.removeHandler(handler)  # 각 핸들러를 제거
        handler.close()  # 핸들러 리소스를 명시적으로 해제
    LOG_FILE="combine.log"    
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    real_total_merge_data = []
    json_num=0
    with open(f"/home/intflow/works/auto_train/cam_id_info.json", 'r') as json_file:
        data = json.load(json_file)
    for key, value in data.items():
        if int(key)<10000:  
            date_type="count"
        elif int(key)>10000 and int(key)<20000: 
            date_type="count"  
        elif int(key)>20000 and int(key)<30000:  
            date_type="grow"
        data_path=value[1]
        
        if os.path.exists(data_path):
            if date_type=="count":
                print("json_path: ",os.path.join(data_path,'json'))
                logging.info("json_path: "+str(os.path.join(data_path,'json')))
                for data_dir in os.listdir(data_path):
                    if os.path.exists(os.path.join((os.path.join(data_path,data_dir)),'json')):
                        json_data_path=os.path.join((os.path.join(data_path,data_dir)),'json')
                        real_total_merge_data.extend(file_list(json_data_path, ext=".json"))
                        print(len(os.listdir(json_data_path)),"장")
                        logging.info(str(len(os.listdir(json_data_path)))+"장")
                        json_num=json_num+len(os.listdir(json_data_path))
            elif date_type=="grow":
                print("json_path: ",os.path.join(data_path,'json'))
                logging.info("json_path: "+str(os.path.join(data_path,'json')))
                if os.path.exists(os.path.join(data_path,'json')):
                    json_data_path=os.path.join(data_path,'json')
                    real_total_merge_data.extend(file_list(json_data_path, ext=".json"))
                    print(len(os.listdir(json_data_path)),"장")
                    logging.info(str(len(os.listdir(json_data_path)))+"장")
                    json_num=json_num+len(os.listdir(json_data_path))
            data_path='/DL_data_super_ssd/new_EFID2023/count/raw_data/unknown_data'
    for data_dir in os.listdir(data_path):
        print("json_path: ", os.path.join((os.path.join(data_path,data_dir)),'json'))
        logging.info("json_path: "+ str(os.path.join((os.path.join(data_path,data_dir)))+'json'))
        if os.path.exists(os.path.join((os.path.join(data_path,data_dir)),'json')):
            json_data_path=os.path.join((os.path.join(data_path,data_dir)),'json')
            real_total_merge_data.extend(file_list(json_data_path, ext=".json"))
            print(len(os.listdir(json_data_path)),"장")
            print(str(len(os.listdir(json_data_path)))+"장")
            json_num=json_num+len(os.listdir(json_data_path))
                # merge_data.extend(file_list("/DL_data_super_ssd/new_EFID2023/grow/raw_data/sungoh_experimental_room/eat2/json", ext=".json"))
    annotation_json_path="/DL_data_super_ssd/new_EFID2023/total/9kpt_combine_total_"+str(json_num)

    
    
    os.makedirs(os.path.join(annotation_json_path,"image"), exist_ok=True)
    os.makedirs(os.path.join(annotation_json_path,"json"), exist_ok=True)
    os.makedirs(os.path.join(annotation_json_path,"drawn_image"), exist_ok=True)
    print(json_num)
    # merge_data.extend(file_list(args.input_json, ext=".json"))
    
    image_list = []
    anno_list = []
    
    total_count = 0
    obj_id = 0
    for each_json in tqdm(real_total_merge_data, total=len(real_total_merge_data) , desc="Processing", unit="it", unit_scale=True):
        new_json_data = load_json(each_json)
        
        # if "cj_vina_unknown" in each_json:
        #     image = cv2.imread(each_json.replace("/single_json/", "/image/").replace(".json", ".jpg"))
        # else:
        source_file =new_json_data['images']['file_name']
        if os.path.exists(new_json_data['images']['file_name']):
            image = cv2.imread(new_json_data['images']['file_name'])
        else:
            print("없음 없음 없음 !!!::",new_json_data['images']['file_name'])
            logging.info("없음 없음 없음 !!!::"+new_json_data['images']['file_name'])
        images_info = {
            "id": total_count,
            "file_name": '{0}.jpg'.format(str(total_count+1).zfill(6)),
            "height": image.shape[0],
            "width": image.shape[1]
        }
        
        image_list.append(images_info)
        
        for value in new_json_data['annotations']:
            anno_json = {
                'id': obj_id,
                'image_id': total_count,
                'category_id': value['category_id'],
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
 
        cv2.imwrite(os.path.join(os.path.join(annotation_json_path,"image"), '{0}.jpg'.format(str(total_count+1).zfill(6))), image)
        
        total_count += 1
    
    categories, poses = set_categories()
    print(total_count)
    train_json = {}
    train_json['images'] = image_list
    train_json['annotations'] = anno_list
    train_json['categories'] = categories
    train_json['poses'] = poses
    with open(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), 'w') as file:
        json.dump(train_json, file, indent=4)
        
    change_anno(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"))
    drawn_image(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), os.path.join(annotation_json_path,"image"), os.path.join(annotation_json_path,"drawn_image"))
    if os.path.exists('combine.log'):
        shutil.move('combine.log', annotation_json_path)
        
def combine_annotion(date_type,farm_list,fod_ok=False):
    # 기존 핸들러 제거 및 새 로그 파일 설정
    logger = logging.getLogger()  # root 로거를 가져옵니다.
    for handler in logger.handlers[:]:  # 기존 핸들러 목록을 복사하며 반복
        logger.removeHandler(handler)  # 각 핸들러를 제거
        handler.close()  # 핸들러 리소스를 명시적으로 해제
    LOG_FILE="combine.log"    
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    with open(f"/home/intflow/works/auto_train/cam_id_info.json", 'r') as json_file:
        data = json.load(json_file)
    json_num=0
    if len(farm_list)==0:
        if date_type=="count":
            data['0']=['unknown_data','/DL_data_super_ssd/new_EFID2023/count/raw_data/unknown_data']
        farm_list=list(data.keys())
    merge_data = []
    for farm_id in farm_list:
        if int(farm_id)<10000:  
            if not date_type=="count":
                continue
        elif int(farm_id)>10000 and int(farm_id)<20000: 
            if not date_type=="count":
                continue  
        elif int(farm_id)>20000 and int(farm_id)<30000:  
            if not date_type=="grow":
                continue
        data_path=data[farm_id][1]
        if os.path.exists(data_path): #count , grow 각자 path 가 다름..
            if date_type=="count":
                for data_dir in os.listdir(data_path):
                    logging.info(("json_path: "+os.path.join(data_path,'json')+" ---> "+ str(len(os.path.join(data_path,'json')))+"장"))
                    print("json_path: ",os.path.join(data_path,'json')," ---> ", len(os.path.join(data_path,'json')),"장")
                    if os.path.exists(os.path.join((os.path.join(data_path,data_dir)),'json')):
                        json_data_path=os.path.join((os.path.join(data_path,data_dir)),'json')
                        merge_data.extend(file_list(json_data_path, ext=".json"))
                        json_num=json_num+len(os.listdir(json_data_path))
            elif date_type=="grow":
                logging.info("json_path: "+os.path.join(data_path,'json')+" ---> "+ str(len(os.path.join(data_path,'json')))+"장")
                print("json_path: ",os.path.join(data_path,'json')," ---> ", len(os.path.join(data_path,'json')),"장")
                if os.path.exists(os.path.join(data_path,'json')):
                    json_data_path=os.path.join(data_path,'json')
                    merge_data.extend(file_list(json_data_path, ext=".json"))
                    json_num=json_num+len(os.listdir(json_data_path))
            # merge_data.extend(file_list("/DL_data_super_ssd/new_EFID2023/grow/raw_data/sungoh_experimental_room/eat2/json", ext=".json"))


    annotation_json_path=""
    if date_type=="count":
        annotation_json_path="/DL_data_super_ssd/new_EFID2023/9kpt_combine_new_hallway_"+str(json_num)
    elif date_type=="grow":
        annotation_json_path="/DL_data_super_ssd/new_EFID2023/grow/training_data/baseline/sungoh_experimental_room_"+str(json_num)
    if fod_ok and date_type=="count": # fod 어노테이션 만들땐 fod 폴더만 보게하려고 한다.
        print("FOD 데이터 폴더를 만들 농장 이름을 작성해주세요 ")
        farm_name=input()
        merge_data = []
        if os.path.exists(os.path.join((os.path.join(data_path,'FOD')),'json')):
            json_data_path=os.path.join((os.path.join(data_path,'FOD')),'json')
            merge_data.extend(file_list(json_data_path, ext=".json"))    
        annotation_json_path=f'/DL_data_super_ssd/new_EFID2023/count/training_data/FOD/{farm_name}'
        print(annotation_json_path)
    
    
    os.makedirs(os.path.join(annotation_json_path,"image"), exist_ok=True)
    os.makedirs(os.path.join(annotation_json_path,"json"), exist_ok=True)
    os.makedirs(os.path.join(annotation_json_path,"drawn_image"), exist_ok=True)
    print(json_num)
    # merge_data.extend(file_list(args.input_json, ext=".json"))
    
    image_list = []
    anno_list = []
    
    total_count = 0
    obj_id = 0
    for each_json in tqdm(merge_data, total=len(merge_data)):
        new_json_data = load_json(each_json)
        
        # if "cj_vina_unknown" in each_json:
        #     image = cv2.imread(each_json.replace("/single_json/", "/image/").replace(".json", ".jpg"))
        # else:
        source_file =new_json_data['images']['file_name']
        if os.path.exists(new_json_data['images']['file_name']):
            image = cv2.imread(new_json_data['images']['file_name'])
        else:
            logging.info("없음 없음 없음 !!!::"+new_json_data['images']['file_name'])
            print("없음 없음 없음 !!!::",new_json_data['images']['file_name'])
        images_info = {
            "id": total_count,
            "file_name": '{0}.jpg'.format(str(total_count+1).zfill(6)),
            "height": image.shape[0],
            "width": image.shape[1]
        }
        
        image_list.append(images_info)
        
        for value in new_json_data['annotations']:
            anno_json = {
                'id': obj_id,
                'image_id': total_count,
                'category_id': value['category_id'],
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
 
        cv2.imwrite(os.path.join(os.path.join(annotation_json_path,"image"), '{0}.jpg'.format(str(total_count+1).zfill(6))), image)
        
        total_count += 1
    
    categories, poses = set_categories()
    print(total_count)
    train_json = {}
    train_json['images'] = image_list
    train_json['annotations'] = anno_list
    train_json['categories'] = categories
    train_json['poses'] = poses
    with open(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), 'w') as file:
        json.dump(train_json, file, indent=4)
        
    change_anno(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"))
    drawn_image(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), os.path.join(annotation_json_path,"image"), os.path.join(annotation_json_path,"drawn_image"))
    if os.path.exists('combine.log'):
        shutil.move('combine.log', annotation_json_path)
        
def combine_eval_annotion():
    # 기존 핸들러 제거 및 새 로그 파일 설정
    logger = logging.getLogger()  # root 로거를 가져옵니다.
    for handler in logger.handlers[:]:  # 기존 핸들러 목록을 복사하며 반복
        logger.removeHandler(handler)  # 각 핸들러를 제거
        handler.close()  # 핸들러 리소스를 명시적으로 해제
    LOG_FILE="combine.log"    
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    from datetime import datetime
    now = datetime.now()
    formatted_date= now.strftime('%Y%m%d')
    real_total_merge_data = []
    for date_type in ['grow','count']: # for문 3개.. 다이어트 필요
        with open(f"/home/intflow/works/auto_train/cam_id_info.json", 'r') as json_file:
            data = json.load(json_file)
            
        total_merge_data = []
        for key, value in data.items():
            merge_data = []
            data_path=value[1]
            json_num=0
            print("data_path: ",data_path)
            if os.path.exists(data_path):
                if date_type=="count":
                    for data_dir in os.listdir(data_path):
                        logging.info("json_path: "+os.path.join((os.path.join(data_path,data_dir))+'json'))
                        print("json_path: ",os.path.join((os.path.join(data_path,data_dir)),'json'))
                        if os.path.exists(os.path.join((os.path.join(data_path,data_dir)),'json')):
                            json_data_path=os.path.join((os.path.join(data_path,data_dir)),'json')
                            random_list=file_random_list(json_data_path, ext=".json")
                            merge_data.extend(random_list)
                            total_merge_data.extend(random_list)
                            real_total_merge_data.extend(random_list)
                            logging.info(str(len(os.listdir(json_data_path))))
                            print(len(os.listdir(json_data_path)))
                            json_num=json_num+len(os.listdir(json_data_path))
                elif date_type=="grow":
                    logging.info("json_path: "+os.path.join(data_path,'json'))
                    print("json_path: ",os.path.join(data_path,'json'))
                    if os.path.exists(os.path.join(data_path,'json')):
                        json_data_path=os.path.join(data_path,'json')
                        random_list=file_random_list(json_data_path, ext=".json")
                        merge_data.extend(random_list)
                        total_merge_data.extend(random_list)
                        real_total_merge_data.extend(random_list)
                        logging.info(str(len(os.listdir(json_data_path))))
                        print(len(os.listdir(json_data_path)))
                        json_num=json_num+len(os.listdir(json_data_path))
                # merge_data.extend(file_list("/DL_data_super_ssd/new_EFID2023/grow/raw_data/sungoh_experimental_room/eat2/json", ext=".json"))



            annotation_json_path=f"/DL_data_super_ssd/new_EFID2023/eval/{formatted_date}/"+str(key)

            
            os.makedirs(f"/DL_data_super_ssd/new_EFID2023/eval/{formatted_date}", exist_ok=True)
            os.makedirs(annotation_json_path, exist_ok=True)
            os.makedirs(os.path.join(annotation_json_path,"image"), exist_ok=True)
            os.makedirs(os.path.join(annotation_json_path,"json"), exist_ok=True)
            os.makedirs(os.path.join(annotation_json_path,"drawn_image"), exist_ok=True)
            print(json_num)
            # merge_data.extend(file_list(args.input_json, ext=".json"))
            
            image_list = []
            anno_list = []
            
            total_count = 0
            obj_id = 0
            for each_json in tqdm(merge_data, total=len(merge_data)):
                new_json_data = load_json(each_json)
                
                # if "cj_vina_unknown" in each_json:
                #     image = cv2.imread(each_json.replace("/single_json/", "/image/").replace(".json", ".jpg"))
                # else:
                source_file =new_json_data['images']['file_name']
                if os.path.exists(new_json_data['images']['file_name']):
                    image = cv2.imread(new_json_data['images']['file_name'])
                else:
                    print("없음 없음 없음 !!!::",new_json_data['images']['file_name'])
                    logging.info("없음 없음 없음 !!!::"+new_json_data['images']['file_name'])
                images_info = {
                    "id": total_count,
                    "file_name": '{0}.jpg'.format(str(total_count+1).zfill(6)),
                    "height": image.shape[0],
                    "width": image.shape[1]
                }
                
                image_list.append(images_info)
                
                for value in new_json_data['annotations']:
                    anno_json = {
                        'id': obj_id,
                        'image_id': total_count,
                        'category_id': value['category_id'],
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
        
                cv2.imwrite(os.path.join(os.path.join(annotation_json_path,"image"), '{0}.jpg'.format(str(total_count+1).zfill(6))), image)
                
                total_count += 1
            
            categories, poses = set_categories()
            print(total_count)
            train_json = {}
            train_json['images'] = image_list
            train_json['annotations'] = anno_list
            train_json['categories'] = categories
            train_json['poses'] = poses
            with open(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), 'w') as file:
                json.dump(train_json, file, indent=4)
                
            change_anno(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"))
            drawn_image(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), os.path.join(annotation_json_path,"image"), os.path.join(annotation_json_path,"drawn_image"))
        image_list = []
        anno_list = []
        
        total_count = 0
        obj_id = 0
        for each_json in tqdm(total_merge_data, total=len(total_merge_data)):
            new_json_data = load_json(each_json)
            annotation_json_path=f"/DL_data_super_ssd/new_EFID2023/eval/{formatted_date}/"+str(date_type)

            
            os.makedirs(f"/DL_data_super_ssd/new_EFID2023/eval/{formatted_date}", exist_ok=True)
            os.makedirs(annotation_json_path, exist_ok=True)
            os.makedirs(os.path.join(annotation_json_path,"image"), exist_ok=True)
            os.makedirs(os.path.join(annotation_json_path,"json"), exist_ok=True)
            os.makedirs(os.path.join(annotation_json_path,"drawn_image"), exist_ok=True)
            # if "cj_vina_unknown" in each_json:
            #     image = cv2.imread(each_json.replace("/single_json/", "/image/").replace(".json", ".jpg"))
            # else:
            source_file =new_json_data['images']['file_name']
            if os.path.exists(new_json_data['images']['file_name']):
                image = cv2.imread(new_json_data['images']['file_name'])
            else:
                print("없음 없음 없음 !!!::",new_json_data['images']['file_name'])
            images_info = {
                "id": total_count,
                "file_name": '{0}.jpg'.format(str(total_count+1).zfill(6)),
                "height": image.shape[0],
                "width": image.shape[1]
            }
            
            image_list.append(images_info)
            
            for value in new_json_data['annotations']:
                anno_json = {
                    'id': obj_id,
                    'image_id': total_count,
                    'category_id': value['category_id'],
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
    
            cv2.imwrite(os.path.join(os.path.join(annotation_json_path,"image"), '{0}.jpg'.format(str(total_count+1).zfill(6))), image)
            
            total_count += 1
        
        categories, poses = set_categories()
        print(total_count)
        train_json = {}
        train_json['images'] = image_list
        train_json['annotations'] = anno_list
        train_json['categories'] = categories
        train_json['poses'] = poses
        with open(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), 'w') as file:
            json.dump(train_json, file, indent=4)
            
        change_anno(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"))        
        drawn_image(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), os.path.join(annotation_json_path,"image"), os.path.join(annotation_json_path,"drawn_image"))
    image_list = []
    anno_list = []
    
    total_count = 0
    obj_id = 0
    for each_json in tqdm(real_total_merge_data, total=len(real_total_merge_data)):
        new_json_data = load_json(each_json)
        annotation_json_path=f"/DL_data_super_ssd/new_EFID2023/eval/{formatted_date}/"+"total"

        
        os.makedirs(f"/DL_data_super_ssd/new_EFID2023/eval/{formatted_date}", exist_ok=True)
        os.makedirs(annotation_json_path, exist_ok=True)
        os.makedirs(os.path.join(annotation_json_path,"image"), exist_ok=True)
        os.makedirs(os.path.join(annotation_json_path,"json"), exist_ok=True)
        os.makedirs(os.path.join(annotation_json_path,"drawn_image"), exist_ok=True)
        # if "cj_vina_unknown" in each_json:
        #     image = cv2.imread(each_json.replace("/single_json/", "/image/").replace(".json", ".jpg"))
        # else:
        source_file =new_json_data['images']['file_name']
        if os.path.exists(new_json_data['images']['file_name']):
            image = cv2.imread(new_json_data['images']['file_name'])
        else:
            print("없음 없음 없음 !!!::",new_json_data['images']['file_name'])
        images_info = {
            "id": total_count,
            "file_name": '{0}.jpg'.format(str(total_count+1).zfill(6)),
            "height": image.shape[0],
            "width": image.shape[1]
        }
        
        image_list.append(images_info)
        
        for value in new_json_data['annotations']:
            anno_json = {
                'id': obj_id,
                'image_id': total_count,
                'category_id': value['category_id'],
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

        cv2.imwrite(os.path.join(os.path.join(annotation_json_path,"image"), '{0}.jpg'.format(str(total_count+1).zfill(6))), image)
        
        total_count += 1
    
    categories, poses = set_categories()
    print(total_count)
    train_json = {}
    train_json['images'] = image_list
    train_json['annotations'] = anno_list
    train_json['categories'] = categories
    train_json['poses'] = poses
    with open(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), 'w') as file:
        json.dump(train_json, file, indent=4)
        
    change_anno(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"))        
    drawn_image(os.path.join(os.path.join(annotation_json_path,"json"),"annotation.json"), os.path.join(annotation_json_path,"image"), os.path.join(annotation_json_path,"drawn_image"))
    if os.path.exists('combine.log'):
        shutil.move('combine.log', annotation_json_path)
        
def split_train_eval_data():
    with open(f"/home/intflow/works/auto_train/cam_id_info.json", 'r') as json_file:
        data = json.load(json_file)
        data['0']=['unknown_data','/DL_data_super_ssd/new_EFID2023/count/raw_data/unknown_data']
        train_merge_data = []
        val_merge_data = []
        for key, value in data.items():
            data_path=value[1]
            if int(key)<10000 and os.path.exists(data_path):
                for data_dir in os.listdir(data_path):
                    logging.info("json_path: "+os.path.join((os.path.join(data_path,data_dir))+'json'))
                    print("json_path: ",os.path.join((os.path.join(data_path,data_dir)),'json'))
                    if os.path.exists(os.path.join((os.path.join(data_path,data_dir)),'json')):
                        json_data_path=os.path.join((os.path.join(data_path,data_dir)),'json')
                        selected_samples,remaining_samples=file_random_split_list(json_data_path, ext=".json")
                        val_merge_data.extend(selected_samples)
                        train_merge_data.extend(remaining_samples)
                        logging.info(str(len(os.listdir(json_data_path))))
                        print(len(os.listdir(json_data_path)))
        annotation_json_path=f"/DL_data_super_ssd/new_EFID2023/count/training_data/baseline/efc_dataset_split_"+str(len(train_merge_data)+len(val_merge_data))

        
        os.makedirs(annotation_json_path, exist_ok=True)
        os.makedirs(os.path.join(annotation_json_path,"image"), exist_ok=True)
        os.makedirs(os.path.join(annotation_json_path,"json"), exist_ok=True)
        os.makedirs(os.path.join(annotation_json_path,"drawn_image"), exist_ok=True)

        total_count = 0
        obj_id = 0
        for merge_data in [train_merge_data,val_merge_data]:
            image_list = []
            anno_list = []
            for each_json in tqdm(merge_data, total=len(merge_data)):
                new_json_data = load_json(each_json)
                

                source_file =new_json_data['images']['file_name']
                if os.path.exists(new_json_data['images']['file_name']):
                    image = cv2.imread(new_json_data['images']['file_name'])
                else:
                    print("없음 없음 없음 !!!::",new_json_data['images']['file_name'])
                    logging.info("없음 없음 없음 !!!::"+new_json_data['images']['file_name'])
                images_info = {
                    "id": total_count,
                    "file_name": '{0}.jpg'.format(str(total_count+1).zfill(6)),
                    "height": image.shape[0],
                    "width": image.shape[1]
                }
                
                image_list.append(images_info)
                
                for value in new_json_data['annotations']:
                    anno_json = {
                        'id': obj_id,
                        'image_id': total_count,
                        'category_id': value['category_id'],
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
                # shutil.copy(new_json_data['images']['file_name'],os.path.join(os.path.join(annotation_json_path,"image"), '{0}.jpg'.format(str(total_count+1).zfill(6))))
                cv2.imwrite(os.path.join(os.path.join(annotation_json_path,"image"), '{0}.jpg'.format(str(total_count+1).zfill(6))), image)
                
                total_count += 1
            
            categories, poses = set_categories()
            print(total_count)
            train_json = {}
            train_json['images'] = image_list
            train_json['annotations'] = anno_list
            train_json['categories'] = categories
            train_json['poses'] = poses
            if merge_data==train_merge_data:
                with open(os.path.join(os.path.join(annotation_json_path,"json"),"annotation_train.json"), 'w') as file:
                    json.dump(train_json, file, indent=4)
                    
                change_anno(os.path.join(os.path.join(annotation_json_path,"json"),"annotation_train.json"))
                drawn_image(os.path.join(os.path.join(annotation_json_path,"json"),"annotation_train.json"), os.path.join(annotation_json_path,"image"), os.path.join(annotation_json_path,"drawn_image"))
            elif merge_data==val_merge_data: 
                with open(os.path.join(os.path.join(annotation_json_path,"json"),"annotation_val.json"), 'w') as file:
                    json.dump(train_json, file, indent=4)
                    
                change_anno(os.path.join(os.path.join(annotation_json_path,"json"),"annotation_val.json"))
                drawn_image(os.path.join(os.path.join(annotation_json_path,"json"),"annotation_val.json"), os.path.join(annotation_json_path,"image"), os.path.join(annotation_json_path,"drawn_image"))

if __name__=="__main__":
    # split_train_eval_data()
    pass