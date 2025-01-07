from segment_anything import SamPredictor, sam_model_registry
# from .fastsam import FastSAM, FastSAMPrompt
from .sam_hq.segment_anything_hq import SamPredictor as SamPredictor_hq
from .sam_hq.segment_anything_hq import sam_model_registry as sam_model_registry_hq
import cv2

from cv2 import (
    BORDER_DEFAULT,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    GaussianBlur,
    morphologyEx,
    getStructuringElement
)
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage import binary_erosion
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

kernel = getStructuringElement(MORPH_ELLIPSE, (8, 8))

class SAMMaskGenerator:
    """
    Segment Anything masking class
    """
    def __init__(self, model=None, ckpt_path=None, is_white_bg=None):
        self.model = model
        self.ckpt_path = ckpt_path
        self.predictor = self.initialize_sam_model(self.model, self.ckpt_path)
        self.is_white_bg = is_white_bg

    def initialize_sam_model(self, model, ckpt_path):
        """
        Initialize SAM model
        """
        if "vit" in model:
            if "hq" in model:
                print("[SAM-HQ] SAM model initialize...")
                vit_model = model[:5]
                sam = sam_model_registry_hq[vit_model](checkpoint=ckpt_path).to(device="cuda:0")
                predictor = SamPredictor_hq(sam)
                print("[SAM-HQ] SAM model loaded!")
            else:
                print("[SAM] SAM model initialize...")
                sam = sam_model_registry[model](checkpoint=ckpt_path).to(device="cuda:0")
                predictor = SamPredictor(sam)
                print("[SAM] SAM model loaded!")
        # elif "FastSAM" in model:
        #     print("[FastSAM] FastSAM model initialize...")
        #     predictor = FastSAM(ckpt_path)
        #     print("[FastSAM] FastSAM model loaded!")
        return predictor

    def generate_mask(self, img, bbox=None):
        if "vit" in self.model:
            if "hq" in self.model:
                # Set image to mask
                print("[SAM-HQ] Loading img to model ")
                self.predictor.set_image(img)
                print("[SAM-HQ] Img loaded!")

                if bbox is None:
                    plt.imshow(img)
                    plt.title('Click on the image')
                    clicked_points_lst = []
                    clicked_points = plt.ginput(1)
                    if clicked_points:
                        for one_click_points in clicked_points:
                            x, y = one_click_points
                            clicked_points_lst.append([int(x), int(y)])
                            # print(f"[SAM] Clicked coordinates: x={x}, y={y}")
                    clicked_points_np = np.array(clicked_points_lst)
                    plt.close()

                    input_point = clicked_points_np
                    input_label = np.array([1])
                    
                    print("[SAM-HQ] Predicting...")
                    time0 = time.time()
                    masks, scores, logits = self.predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True
                        )
                    time1 = time.time()
                    print(f"[SAM-HQ] cpu time : {time1 - time0}")
                    self.show_all_mask_and_score(masks, scores, img)
                    selected_score_idx = input("[SAM] Input your idx : ")
                    final_mask = masks[int(selected_score_idx) - 1]
                else:        
                    masks, scores, logits = self.predictor.predict(
                        box=bbox,
                        multimask_output=False,
                        
                    )
                    max_score_idx = np.argmax(scores)
                    print(f"score : {scores[max_score_idx]}")
                    final_mask = masks[max_score_idx]
            else:
                # Set image to mask
                print("[SAM] Loading img to model ")
                self.predictor.set_image(img)
                print("[SAM] Img loaded!")

                if bbox is None:
                    plt.imshow(img) # x=391, y=223
                    plt.title('Click on the image')
                    clicked_points = plt.ginput(1)
                    if clicked_points:
                        x, y = clicked_points[0]
                        print(f"[SAM] Clicked coordinates: x={x}, y={y}")
                    plt.close()
                    # Set object mask query point
                    x_point = int(x)
                    y_point = int(y)

                    input_point = np.array([[x_point, y_point]])
                    input_label = np.array([1])
                    
                    print("[SAM] Predicting...")
                    time0 = time.time()
                    masks, scores, logits = self.predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True
                        )
                    time1 = time.time()
                    print(f"[SAM] cpu time : {time1 - time0}")

                    self.show_all_mask_and_score(masks, scores, img)
                    selected_score_idx = input("[SAM] Input your idx : ")
                    final_mask = masks[int(selected_score_idx) - 1]
                else:        
                    masks, scores, logits = self.predictor.predict(
                        box=bbox,
                        multimask_output=True
                    )
                    max_score_idx = np.argmax(scores)
                    print(f"score : {scores[max_score_idx]}")
                    final_mask = masks[max_score_idx]            
        # elif "FastSAM" in self.model:
        #     if bbox is not None:
        #         bbox = bbox.tolist()
        #         print(f"[FastSAM] inferencing...")
        #         time0 = time.time()
        #         everything_results = self.predictor(
        #             img,
        #             device="cuda:0",
        #             retina_masks=True,
        #             imgsz=1024,
        #             conf=0.4,
        #             iou=0.9,
        #             verbose=False
        #         )
        #         prompt_process = FastSAMPrompt(img, everything_results, device="cuda:0")
        #         final_mask = prompt_process.box_prompt(bbox=bbox) # (1, H, W)
        #         print(f"[FastSAM] Done! ({time.time() - time0}sec)")

        final_mask = final_mask.astype(np.uint8).squeeze()

        # Mask post processing
        processed_mask = self.post_process_mask(final_mask)
        processed_mask = Image.fromarray(processed_mask)
        img = Image.fromarray(img)
        processed_img = self.alpha_matting_cutout(img,
                                                  processed_mask,
                                                  240,
                                                  10,
                                                  10)
        if "vit" in self.model or "hq" in self.model:
            self.predictor.reset_image()
        return np.asarray(processed_img), np.asarray(processed_mask), np.asarray(img)


    def show_mask(self, mask, random_color=False):
        color = np.array([255/255, 1/255, 1/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.imshow(mask_image)

    def show_all_mask_and_score(self, masks, scores, img):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(img)
            self.show_mask(mask, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()  

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewsidth=1.25)  

    def post_process_mask(self, mask):
        """
        Post Process the mask for a smooth boundary by applying Morphological Operations
        Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
        args:
            mask: Binary Numpy Mask
        """
        mask = morphologyEx(mask, MORPH_OPEN, kernel)
        mask = GaussianBlur(mask, (3, 3), sigmaX=3, sigmaY=3, borderType=BORDER_DEFAULT)
        mask = (mask * 255).astype(np.uint8)
        return mask
    
    def alpha_matting_cutout(
        self,
        img: PILImage,
        mask: PILImage,
        foreground_threshold: int,
        background_threshold: int,
        erode_structure_size: int,
    ) -> PILImage:
        """
        Perform alpha matting on an image using a given mask and threshold values.

        This function takes a PIL image `img` and a PIL image `mask` as input, along with
        the `foreground_threshold` and `background_threshold` values used to determine
        foreground and background pixels. The `erode_structure_size` parameter specifies
        the size of the erosion structure to be applied to the mask.

        The function returns a PIL image representing the cutout of the foreground object
        from the original image.
        """
        if img.mode == "RGBA" or img.mode == "CMYK":
            img = img.convert("RGB")

        img = np.asarray(img)
        mask = np.asarray(mask)

        is_foreground = mask > foreground_threshold
        is_background = mask < background_threshold

        structure = None
        if erode_structure_size > 0:
            structure = np.ones(
                (erode_structure_size, erode_structure_size), dtype=np.uint8
            )

        is_foreground = binary_erosion(is_foreground, structure=structure)
        is_background = binary_erosion(is_background, structure=structure, border_value=1)

        trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
        trimap[is_foreground] = 255
        trimap[is_background] = 0

        img_normalized = img / 255.0
        trimap_normalized = trimap / 255.0

        alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
        foreground = estimate_foreground_ml(img_normalized, alpha)
        cutout = stack_images(foreground, alpha)

        cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
        cutout = Image.fromarray(cutout)

        return cutout