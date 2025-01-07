from groundingdino.util.inference import load_model, load_image, predict, annotate
import matplotlib.pyplot as plt
import os
import glob
import time
import cv2

def get_dino_model(dino_config_path,
                   dino_weight_path,
                   device):
    model = load_model(model_config_path=dino_config_path,
                       model_checkpoint_path=dino_weight_path,
                       device=device)
    print(f"[DINO] model loaded (mode : {device})")
    return model

def get_annotation(model, image_path, dino_params):
    TEXT_PROMPT   = dino_params["text_prompt"]
    BOX_TRESHOLD  = dino_params["box_threshold"]
    TEXT_TRESHOLD = dino_params["text_threshold"]

    print(f"[DINO] predicting ...")
    time0 = time.time()
    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device="cpu"
    )
    annotated_frame, xyxy = annotate(image_source=image_source,
                                    boxes=boxes,
                                    logits=logits,
                                    phrases=phrases)
    print(f"[DINO] done! : {time.time() - time0:.2f}s")
    return annotated_frame, xyxy