from sam_util.generate_mask import SAMMaskGenerator
from sam_util.dino_detector import get_dino_model, get_annotation
from ultralytics import YOLO

import glob
import os
import cv2
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm

detection_model = "yolo" # yolo, dino
seg_model = "sam_hq" # vanilla_sam, sam_hq
verbose = False

object_name = "toycar" # bulldozer, green_stapler, screw, bottle, toycar, mouse
on_gripper = True

is_remask = False
re_mask = []

print(f"is verbose : {verbose}")
print(f"on_gripper : {on_gripper}")
print(f"is_remask : {is_remask}")

# Model initialize
if detection_model == "yolo":
    if on_gripper is True:
        object_name = f"{object_name}_gripper"
        path = glob.glob("pre_generated_data/image_obj_cam2/*.png")
    else:
        path = glob.glob("pre_generated_data/image_obj/*.png")
    
    print("[YOLO] yolo model load")
    yolo_model = YOLO(f"sam_util/runs_{object_name}/detect/train/weights/best.pt")
    yolo_model.to("cpu")
    print("[YOLO] yolo model loaded")

elif detection_model == "dino":
    dino_config_path = os.path.join("sam_util",
                                    "GroundingDINO",
                                    "groundingdino",
                                    "config",
                                    "GroundingDINO_SwinT_OGC.py")
    dino_weight_path = os.path.join("sam_util",
                                    "GroundingDINO",
                                    "weights",
                                    "groundingdino_swint_ogc.pth")
    device = "cpu"
    dino_model = get_dino_model(dino_config_path,
                                dino_weight_path,
                                device)

    text_prompt    = "text prompt for your object" # ex. green stapler in white table
    box_threshold  = 0.35
    text_threshold = 0.25

    dino_params = {
        "text_prompt"    : text_prompt,
        "box_threshold"  : box_threshold,
        "text_threshold" : text_threshold 
    }
    path = glob.glob("pre_generated_data/image_obj/*.png")

if seg_model == "vanilla_sam":
    ## For vanila SAM
    sam_model = "vit_h"
    is_white = False
    ckpt_path = "sam_util/sam_vit_h_4b8939.pth"

elif seg_model == "sam_hq":
    ## For SAM-HQ
    model_size = "h"
    sam_model = f"vit_{model_size}_hq"
    is_white = False
    ckpt_path = f"sam_util/sam_hq/pretrained_checkpoint/sam_hq_vit_{model_size}.pth"



sam_generator = SAMMaskGenerator(sam_model, ckpt_path, is_white)

# Path initialize
original_img_path = os.path.join("pre_generated_data", "image")
mask_path = os.path.join("pre_generated_data", "mask")
rmbg_path = os.path.join("pre_generated_data", "image_rmbg")

os.makedirs(mask_path, exist_ok=True)
os.makedirs(rmbg_path, exist_ok=True)

path = sorted(path)

if not is_remask:
    # GOGO
    for one_path in tqdm(path):
        number = re.search(r'(\d+).png$', one_path).group(1)
        current_image = cv2.imread(one_path)
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        if verbose:
            plt.imshow(current_image)
            plt.show()
            plt.close()
        if detection_model == "yolo":
            ## For YOLO Detection    
            print(number)
            detection_result = yolo_model.predict(one_path, device="cpu", verbose=False)
            res_plot = detection_result[0].plot()
            bbox_xyxy = detection_result[0].boxes[0].xyxy.numpy().astype(int).squeeze()
            if verbose:
                print(f"[YOLO] BBOX : {bbox_xyxy}")
                plt.imshow(res_plot)
                plt.show()
                plt.close()
        elif detection_model == "dino":
            ## For DINO detection
            annotated_frame, bbox_xyxy = get_annotation(model=dino_model,
                                                image_path=one_path,
                                                dino_params=dino_params)
            if verbose:
                print(f"[DINO] BBOX : {bbox_xyxy}")
                plt.imshow(annotated_frame)
                plt.show()
                plt.close()
        try:
            processed_img, processed_mask, original_img = sam_generator.generate_mask(current_image, bbox=bbox_xyxy)
            if verbose:
                plt.imshow(processed_img)
                plt.show()
                plt.close()
            
            cv2.imwrite(os.path.join(rmbg_path, f"{number}.png"), cv2.cvtColor(processed_img, cv2.COLOR_BGRA2RGBA))
            cv2.imwrite(os.path.join(mask_path, f"{number}.png"), processed_mask)
        except:
            print(f" >>>>>>>>>>> cannot get mask : {one_path}")
            pass
else:
    for one_re_mask in re_mask:
        print(one_re_mask)
        if one_re_mask < 100 and one_re_mask >= 10:
            if detection_model == "dino":
                one_path = f"pre_generated_data/image_obj/0{one_re_mask}.png"
            else:
                one_path = f"pre_generated_data/image_obj_cam2/0{one_re_mask}.png"

        else:
            if detection_model == "dino":
                one_path = f"pre_generated_data/image_obj/{one_re_mask}.png"
            else:
                one_path = f"pre_generated_data/image_obj_cam2/{one_re_mask}.png"
        # number = re.search(r'(\d+).$', one_path).group(1)
        current_image = cv2.imread(one_path)
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        processed_img, processed_mask, original_img = sam_generator.generate_mask(current_image)
        cv2.imwrite(os.path.join(rmbg_path, f"{one_re_mask}.png"), cv2.cvtColor(processed_img, cv2.COLOR_BGRA2RGBA))
        cv2.imwrite(os.path.join(mask_path, f"{one_re_mask}.png"), processed_mask)
        

