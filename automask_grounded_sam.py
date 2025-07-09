import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.path.join(project_root, "Grounded-Sam-2")

GROUNDING_MODEL = os.path.join(project_root, "models/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71")
TEXT_PROMPT = "stair."
IMG_PATH = "my_images/image.png"
SAM2_CHECKPOINT = os.path.join(project_root, "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
# SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
# SAM2_MODEL_CONFIG = os.path.join(project_root,"Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"
OUTPUT_DIR = Path("outputs/grounded_sam2_hf_demo")
DUMP_JSON_RESULTS = False

class GroundedSAM2:
    def __init__(self, grounding_model=GROUNDING_MODEL, img_path=IMG_PATH, sam2_checkpoint=SAM2_CHECKPOINT, 
                 sam2_model_config=SAM2_MODEL_CONFIG, output_dir=OUTPUT_DIR, no_dump_json=DUMP_JSON_RESULTS, force_cpu=True):
        self.grounding_model = grounding_model
        self.text_prompt = None
        self.img_path = img_path
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_config = sam2_model_config
        self.output_dir = Path(output_dir)
        self.dump_json_results = not no_dump_json
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"

        # create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # environment settings
        # use bfloat16
        torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # build SAM2 image predictor
        sam2_checkpoint = SAM2_CHECKPOINT
        model_cfg = SAM2_MODEL_CONFIG
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino from huggingface
        model_id = self.grounding_model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

        
    def call_grounding_dino(self):
        text = self.text_prompt
        # img_path = self.img_path
        image = self.img
        # self.sam2_predictor.set_image(np.array(image.convert("RGB")))
        self.sam2_predictor.set_image(np.array(image.convert("RGB")))
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        return results
    
    def call_sam2_predictor(self,dino_results):
        # get the box prompt for SAM 2
        input_boxes = dino_results[0]["boxes"].cpu().numpy()
        # print(f"input_boxes shape: {input_boxes.shape}")
        # print(f"input_boxes: {input_boxes}")
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        return masks

    def generate_annotated_frame(self, dino_results, sam2_masks):
        masks = sam2_masks
        results = dino_results
        # image = Image.open(self.img_path)
        image = self.img
        input_boxes = results[0]["boxes"].cpu().numpy()
        #print(f"input_boxes:{input_boxes}")
        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        """
        Visualize image with supervision useful API
        """
        # img = cv2.imread(self.img_path)
        img = self.img
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )
        """
        Note that if you want to use default color map,
        you can set color=ColorPalette.DEFAULT
        """
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        # groundingdino_annotated_image
        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        # cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)
        # grounded_sam2_annotated_image_with_mask
        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame_with_mask = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        # cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

        return annotated_frame_with_mask

    # def dump_results(self,dino_results,sam2_masks):
    #     masks = sam2_masks
    #     results = dino_results
    #     # image = Image.open(self.img_path)
    #     image = self.img
    #     input_boxes = results[0]["boxes"].cpu().numpy()
    #     """
    #     Post-process the output of the model to get the masks, scores, and logits for visualization
    #     """
    #     # convert the shape to (n, H, W)
    #     if masks.ndim == 4:
    #         masks = masks.squeeze(1)
    #     confidences = results[0]["scores"].cpu().numpy().tolist()
    #     class_names = results[0]["labels"]
    #     class_ids = np.array(list(range(len(class_names))))
    #     labels = [
    #         f"{class_name} {confidence:.2f}"
    #         for class_name, confidence
    #         in zip(class_names, confidences)
    #     ]
    #     """
    #     Visualize image with supervision useful API
    #     """
    #     # img = cv2.imread(self.img_path)
    #     img = self.img
    #     detections = sv.Detections(
    #         xyxy=input_boxes,  # (n, 4)
    #         mask=masks.astype(bool),  # (n, h, w)
    #         class_id=class_ids
    #     )
    #     """
    #     Note that if you want to use default color map,
    #     you can set color=ColorPalette.DEFAULT
    #     """
    #     box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    #     annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    #     label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    #     annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    #     cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)
    #     mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    #     annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    #     cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)
    #     """
    #     Dump the results in standard format and save as json files
    #     """
    #     def single_mask_to_rle(mask):
    #         rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    #         rle["counts"] = rle["counts"].decode("utf-8")
    #         return rle
    #     if DUMP_JSON_RESULTS:
    #         # convert mask into rle format
    #         mask_rles = [single_mask_to_rle(mask) for mask in masks]
    #         input_boxes = input_boxes.tolist()
    #         scores = scores.tolist()
    #         # save the results in standard format
    #         results = {
    #             "image_path": self.img_path,
    #             "annotations" : [
    #                 {
    #                     "class_name": class_name,
    #                     "bbox": box,
    #                     "segmentation": mask_rle,
    #                     "score": score,
    #                 }
    #                 for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
    #             ],
    #             "box_format": "xyxy",
    #             "img_width": image.width,
    #             "img_height": image.height,
    #         }
            
    #         with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
    #             json.dump(results, f, indent=4)

    def generate_mask(self,image,text):
        self.text_prompt=text
        self.img = Image.fromarray(image)
        dino_results = self.call_grounding_dino()
        if dino_results[0]["boxes"].shape[0]==0:
            return None,None
        sam2_masks = self.call_sam2_predictor(dino_results)
        # self.dump_results(dino_results, sam2_masks)
        # input_boxes = dino_results[0]["boxes"].cpu().numpy()
        return dino_results, sam2_masks

if __name__ == "__main__":
    grounede_sam2 = GroundedSAM2()
    grounede_sam2.generate_mask()