from pickle import FALSE
import torch 
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torchvision
from zipfile import ZipFile 
import os
import multiprocessing
import math
import numpy as np
import random 
<<<<<<< HEAD
import cv2
=======

>>>>>>> 121384c21c7be07787f3bd89302b043b247ec57e

VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.tiff', '.bmp', '.png']


def check_filenames_in_zipdata(filenames, ziproot):
    samples = []
    for fst in ZipFile(ziproot).infolist():
        fname = fst.filename
        if fname.endswith('/') or fname.startswith('.') or fst.file_size == 0:
            continue
        if os.path.splitext(fname)[1].lower() in VALID_IMAGE_TYPES:
            samples.append((fname))
    filenames = set(filenames)
    samples = set(samples)
    assert filenames.issubset(samples), 'Something wrong with your zip data'

def rotated_rectangle_to_polygon(cx, cy, w, h, theta, W, H):
    """Convert a rotated rectangle specified by a center point, width, height, and rotation (in radians)
    into the four corners of the rectangle."""
    cx *= W
    cy *= H
    w *= W
    h *= H

    dx = w / 2
<<<<<<< HEAD
    dy = h / 2 
=======
    dy = h / 2
>>>>>>> 121384c21c7be07787f3bd89302b043b247ec57e

    cos_angle = math.cos(theta)
    sin_angle = math.sin(theta)

    # Compute the four corners in clockwise order
    corners = []
    for corner in [(dx, dy), (-dx, dy), (-dx, -dy), (dx, -dy)]:
        x, y = corner
        corners.append((cx + (x * cos_angle - y * sin_angle), cy + (x * sin_angle + y * cos_angle)))

    return sum(corners, ())  # Flatten the list of tuples


def draw_rbbox(img, rbboxes, W, H):
<<<<<<< HEAD
    colors = [(0, 0, 255), (0, 128, 128), (255, 0, 0), (0, 255, 0),
              (0, 165, 255), (42, 42, 165), (255, 255, 0), (128, 0, 128)]
    
    for bid, rbbox in enumerate(rbboxes):
        cx, cy, w, h, theta = rbbox
        corners = get_xy_point([cx, cy, w, h, theta], W, H)
        corners = [np.array(corners).flatten().tolist()]
        corners_rbbox = np.array(corners).astype(int).reshape(4,2)
        color = colors[2]
        cv2.drawContours(img, [corners_rbbox], 0, color=color, thickness=3)
    return img


def get_xy_point(rbbox, W, H):

    cx, cy, width, height, radian = [i.cpu().item() for i in rbbox]
    if height > width:
        print(f"ERROR")
    cx *= W
    cy *= H
    width *= W
    height *= H
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

=======
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    for bid, rbbox in enumerate(rbboxes):
        cx, cy, w, h, theta = rbbox
        box = rotated_rectangle_to_polygon(cx, cy, w, h, theta, W, H)
        draw.polygon(box, outline=colors[bid % len(colors)], width=4)
    return img

>>>>>>> 121384c21c7be07787f3bd89302b043b247ec57e
def draw_box(img, boxes):
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline =colors[bid % len(colors)], width=4)
        # draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 


def draw_points(img, points, W, H):
    N, K = points.shape
<<<<<<< HEAD
    num_kp = int(K / 2)
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0),
              (0, 165, 255), (42, 42, 165), (255, 255, 0), (128, 0, 128),
              (255, 20, 147), (255, 127, 80), (255, 215, 0), (0, 0, 139),
              (240, 230, 140), (144, 238, 144), (255, 250, 250), (154, 205, 50),
              (50, 205, 50)]
    colors = colors[:num_kp]
    
    for point in points:
        for k in range(num_kp):
            if point[2*k] == point[2*k+1] == 0:
                continue
            x, y = int(point[2*k]*W), int(point[2*k+1]*H)
            cv2.circle(img, (x, y), 3, colors[k], -1)  # -1 fills the circle
    return img

=======
    num_kp = int(K*0.5)
    colors = ["red", "yellow", "blue", "green", "orange", "brown", "cyan", "purple", "deeppink", "coral", "gold", "darkblue", "khaki", "lightgreen", "snow", "yellowgreen", "lime"]
    colors = colors[:num_kp]
    draw = ImageDraw.Draw(img)
    
    r = 3
    for point in points:
        for k in range(0,num_kp):
            if point[2*k] == point[2*k+1] == 0:
                pass 
            else:
                x, y = float(point[2*k]*W), float(point[2*k+1]*H)
                draw.ellipse( [ (x-r,y-r), (x+r,y+r) ], fill=colors[k])
    return img 
>>>>>>> 121384c21c7be07787f3bd89302b043b247ec57e



def to_valid(x0, y0, x1, y1, kps, image_size, min_box_size):
    valid = True

    # ---------------- check if box still exist ------------------- # 
    if x0>image_size or y0>image_size or x1<0 or y1<0:
        valid = False # no way to make this box vide, it is completely cropped out 
        return valid, (None,None,None,None), None


    # ---------------- check if box too small ------------------- # 
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_size)
    y1 = min(y1, image_size)
    if (x1-x0)*(y1-y0) / (image_size*image_size) < min_box_size:
        valid = False
        return valid, (None,None,None,None), None


    # ---------------- check if all pts exists ------------------- # 
    for kp in kps:
        if kp["valid"]:
            kp_x, kp_y = kp["loc"] 
            if kp_x<0 or kp_x>image_size or kp_y<0 or kp_y>image_size: # this kp was cropped out
                kp['valid'] = False
                kp["loc"] = [0,0]

    if all([ not kp["valid"] for kp in kps  ]):
        valid = False # all kps were cropped but box is still valid (It's unlikely though)
        return valid, (None,None,None,None), None


    return valid, (x0, y0, x1, y1), kps

def to_valid_rbbox(scaled_cx, scaled_cy, scaled_w, scaled_h, adjusted_angle, kps, image_size, min_box_size):
    valid = True

    # Check if the center of the box is still within the image
    if not (0 <= scaled_cx <= image_size and 0 <= scaled_cy <= image_size):
<<<<<<< HEAD
        return False, (None, None, None, None, None), None

    # Check if the box is too small
    if scaled_w * scaled_h / (image_size * image_size) < min_box_size:
        return False, (None, None, None, None, None), None
=======
        return False, (None, None, None, None, None)

    # Check if the box is too small
    if scaled_w * scaled_h / (image_size * image_size) < min_box_size:
        return False, (None, None, None, None, None)
>>>>>>> 121384c21c7be07787f3bd89302b043b247ec57e

    # ---------------- check if all pts exists ------------------- # 
    for kp in kps:
        if kp["valid"]:
            kp_x, kp_y = kp["loc"] 
            if kp_x<0 or kp_x>image_size or kp_y<0 or kp_y>image_size: # this kp was cropped out
                kp['valid'] = False
                kp["loc"] = [0,0]

    if all([ not kp["valid"] for kp in kps  ]):
        valid = False # all kps were cropped but box is still valid (It's unlikely though)
        return valid, (None,None,None,None, None), None


    return valid, (scaled_cx, scaled_cy, scaled_w, scaled_h, adjusted_angle), kps



def recalculate_box_kps_and_verify_if_valid(x, y, w, h, kps, trans_info, image_size, min_box_size):
    """
    box [x,y,w,h]:  the original annotation corresponding to the raw image size.
    kpts: the origianl labled visible kpts 
    trans_info: what resizing and cropping have been applied to the raw image 
    image_size:  what is the final image size  
    """
    x0 = x * trans_info["performed_scale"] - trans_info['crop_x'] 
    y0 = y * trans_info["performed_scale"] - trans_info['crop_y'] 
    x1 = (x + w) * trans_info["performed_scale"] - trans_info['crop_x'] 
    y1 = (y + h) * trans_info["performed_scale"] - trans_info['crop_y'] 


    for kp in kps:
        if kp["valid"]:
            kp_x, kp_y = kp["loc"] 
            kp_x = kp_x * trans_info["performed_scale"] - trans_info['crop_x']
            kp_y = kp_y * trans_info["performed_scale"] - trans_info['crop_y'] 
            kp["loc"] = [kp_x, kp_y]
               

    # at this point, box annotation has been recalculated based on scaling and cropping
    # but some point may fall off the image_size region (e.g., negative value), thus we 
    # need to clamp them into 0-image_size. But if all points falling outsize of image 
    # region, then we will consider this is an invalid box. 
    valid, (x0, y0, x1, y1), kps = to_valid(x0, y0, x1, y1, kps, image_size, min_box_size)

    if valid:
        # we also perform random flip. 
        # Here boxes are valid, and are based on image_size 
        if trans_info["performed_flip"]:
            x0, x1 = image_size-x1, image_size-x0
            for kp in kps:
                if kp["valid"]:
                    kp_x, kp_y = kp["loc"] 
                    kp["loc"] = [image_size-kp_x, kp_y]

    return valid, (x0, y0, x1, y1), kps


def recalculate_rbbox_kps_and_verify_if_valid(cx, cy, w, h, angle, kps, trans_info, image_size, min_box_size):
    """
    rbbox [cx,cy,w,h, angle]:  the original annotation corresponding to the raw image size.
    kpts: the origianl labled visible kpts 
    trans_info: what resizing and cropping have been applied to the raw image 
    image_size:  what is the final image size  
    """
    # Scale the center point
    scaled_cx = cx * trans_info["performed_scale"] - trans_info['crop_x']
    scaled_cy = cy * trans_info["performed_scale"] - trans_info['crop_y']
    
    # Scale width and height
    scaled_w = w * trans_info["performed_scale"]
    scaled_h = h * trans_info["performed_scale"]

    # Adjust angle for horizontal flip
    if trans_info["performed_flip"]:
        adjusted_angle = -angle  # Assuming angle in degrees
    else:
        adjusted_angle = angle

    for kp in kps:
        if kp["valid"]:
            kp_x, kp_y = kp["loc"] 
            kp_x = kp_x * trans_info["performed_scale"] - trans_info['crop_x']
            kp_y = kp_y * trans_info["performed_scale"] - trans_info['crop_y'] 
            kp["loc"] = [kp_x, kp_y]
               

    # at this point, box annotation has been recalculated based on scaling and cropping
    # but some point may fall off the image_size region (e.g., negative value), thus we 
    # need to clamp them into 0-image_size. But if all points falling outsize of image 
    # region, then we will consider this is an invalid box. 
    valid, (cx, cy, w, h, angle), kps = to_valid_rbbox(scaled_cx, scaled_cy, scaled_w, scaled_h, adjusted_angle, kps, image_size, min_box_size)
    

    # CHECK THE PERFORMED FLIP RESULT
    if valid and trans_info["performed_flip"]:
        # Mirror the center x-coordinate
        flipped_cx = image_size - scaled_cx

        # Adjust the angle for the flip
        # Assuming angle increases clockwise and 0 degrees is rightward
        flipped_angle = (360 - adjusted_angle) % 360

        # Update keypoints if present
        for kp in kps:
            if kp["valid"]:
                kp_x, kp_y = kp["loc"]
                kp["loc"] = [image_size - kp_x, kp_y]  # Mirror the x-coordinate of keypoints

        # Ensure to update the rbbox parameters with the flipped ones
        cx, angle = flipped_cx, flipped_angle

    return valid, (cx, cy, w, h, angle), kps




class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, random_crop, random_flip, image_size):
        super().__init__() 
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.image_size = image_size
        self.zip_dict = {}

        if self.random_crop:
            assert False, 'NOT IMPLEMENTED'


    def fetch_zipfile(self, ziproot):
        pid = multiprocessing.current_process().pid # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipFile(ziproot)
        zip_file = self.zip_dict[pid]
        return zip_file

<<<<<<< HEAD
    # def vis_getitem_data(self, index=None, out=None, return_tensor=False, name="res.jpg", print_caption=True):
    
    #     if out is None:
    #         out = self[index]

    #     img = torchvision.transforms.functional.to_pil_image( out["image"]*0.5+0.5 )
    #     canvas = torchvision.transforms.functional.to_pil_image( torch.ones_like(out["image"]) )
    #     W, H = img.size

    #     if print_caption:
    #         caption = out["caption"]
    #         print(caption)
    #         print(" ")

    #     # boxes = []
    #     # for box in out["boxes"]:    
    #     #     x0,y0,x1,y1 = box
    #     #     boxes.append( [float(x0*W), float(y0*H), float(x1*W), float(y1*H)] )
    #     # img = draw_box(img, boxes)
    #     # img = draw_points( img, out["points"], W, H )   
    #     rbboxes = []
    #     for rbbox in out["rbboxes"]:
    #         cx, cy, w, h, theta = rbbox
    #         rbboxes.append([cx, cy, w, h, theta])
    #     img = draw_rbbox(img, rbboxes, W, H)
    #     img = draw_points( img, out["points"], W, H )   

    #     if return_tensor:
    #         return  torchvision.transforms.functional.to_tensor(img)
    #     else:
    #         img.save(name)  

    def vis_getitem_data(self, index=None, out=None, return_tensor=False, name="res.jpg", print_caption=True):
        if out is None:
            out = self[index]
        
        # Convert the PyTorch tensor to a NumPy array and adjust color channel order
        img_np = ((out["image"]*0.5+0.5).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
=======
    def vis_getitem_data(self, index=None, out=None, return_tensor=False, name="res.jpg", print_caption=True):
    
        if out is None:
            out = self[index]

        img = torchvision.transforms.functional.to_pil_image( out["image"]*0.5+0.5 )
        canvas = torchvision.transforms.functional.to_pil_image( torch.ones_like(out["image"]) )
        W, H = img.size

>>>>>>> 121384c21c7be07787f3bd89302b043b247ec57e
        if print_caption:
            caption = out["caption"]
            print(caption)
            print(" ")

<<<<<<< HEAD
        # Assuming W, H can be derived directly from the image tensor
        H, W = img_np.shape[:2]
        
        rbboxes = [rbbox for rbbox in out["rbboxes"]]
        img_np = draw_rbbox(img_np, rbboxes, W, H)
        
        points = out["points"]
        img_np = draw_points(img_np, points, W, H)

        if return_tensor:
            # Convert back to tensor
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        else:
            cv2.imwrite(name, img_np)
=======
        # boxes = []
        # for box in out["boxes"]:    
        #     x0,y0,x1,y1 = box
        #     boxes.append( [float(x0*W), float(y0*H), float(x1*W), float(y1*H)] )
        # img = draw_box(img, boxes)
        # img = draw_points( img, out["points"], W, H )   
        rbboxes = []
        for rbbox in out["rbbox"]:
            cx, cy, w, h, theta = rbbox
            rbboxes.append([cx, cy, w, h, theta])
        img = draw_rbbox(img, rbboxes, W, H)
        img = draw_points( img, out["points"], W, H )   

        if return_tensor:
            return  torchvision.transforms.functional.to_tensor(img)
        else:
            img.save(name)  
>>>>>>> 121384c21c7be07787f3bd89302b043b247ec57e

    def transform_image(self, pil_image):
        if self.random_crop:
            assert False
            arr = random_crop_arr(pil_image, self.image_size) 
        else:
            arr, info = center_crop_arr(pil_image, self.image_size)
		
        info["performed_flip"] = False
        if self.random_flip and random.random()<0.5:
            arr = arr[:, ::-1]
            info["performed_flip"] = True
		
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2,0,1])

        return torch.tensor(arr), info 



def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    WW, HH = pil_image.size

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)

    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # at this point, the min of pil_image side is desired image_size
    performed_scale = image_size / min(WW, HH)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    
    info = {"performed_scale":performed_scale, 'crop_y':crop_y, 'crop_x':crop_x, "WW":WW, 'HH':HH}

    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size], info


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]