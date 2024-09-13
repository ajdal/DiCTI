"""
patches = {"1":"Torso", "2":"Torso", "3":"Right Hand", "4":"Left Hand", "5":"Left Foot", "6":"Right Foot", "7":"Upper Leg Right", "9":"Upper Leg Right",
   "8":"Upper Leg Left", "10":"Upper Leg Left", "11":"Lower Leg Right", "13":"Lower Leg Right", "12":"Lower Leg Left", "14":"Lower Leg Left",
   "15":"Upper Arm Left", "17":"Upper Arm Left", "16":"Upper Arm Right", "18":"Upper Arm Right", "19":"Lower Arm Left", "21":"Lower Arm Left",
   "20":"Lower Arm Right", "22":"Lower Arm Right", "23":"Head", "24":"Head"}
"""

import os
from typing import List, Union, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector

# Set to True to save intermediate results for debugging
DEBUG = False
DEBUG_OUTPUT_PATH = "output/tmp"



def square_pad(img, bg):
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), bg)
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), bg)
        result.paste(img, ((height - width) // 2, 0))
        return result


def crop_top_left(img, bg=None):
    width, height = img.size
    min_dim = min(width, height)
    if width == height:
        return img
    else:
        return img.crop((0, 0, min_dim, min_dim))


def seg_from_densepose(dp, img_size):
    seg = Image.new("P", img_size, color=0)
    i = torch.argmax(dp["scores"]).item()
    xmin, ymin, _, _ = dp["pred_boxes_XYXY"][i]
    xmin, ymin = round(xmin.item()), round(ymin.item())
    labels = Image.fromarray(
        dp["pred_densepose"][0].labels.to("cpu").to(torch.uint8).numpy()
    ).convert("P")
    seg.paste(labels, (xmin, ymin))
    seg = np.asarray(seg)
    return seg


def load_data(img_path, dp_path):
    img = Image.open(img_path).convert("RGB")
    with open(dp_path, "rb") as fp:
        dp = torch.load(fp)
    seg = seg_from_densepose(dp, img.size)
    return img, seg


def generate_masks(seg, mask_padding=70, mask_erosion=3, head_erosion=5):
    seg_body = np.logical_and(1 <= seg, seg <= 22)
    seg_head = seg >= 23
    seg_head_hands_feet = np.logical_or(np.logical_and(3 <= seg, seg <= 6), seg >= 23)
    seg_body_hands_feet = np.logical_or(
        np.logical_and(1 <= seg, seg <= 2), np.logical_and(7 <= seg, seg <= 22)
    )

    if DEBUG:
        Image.fromarray(seg_body).save(os.path.join(DEBUG_OUTPUT_PATH, "body_segx.png"))
        Image.fromarray(seg_head).save(os.path.join(DEBUG_OUTPUT_PATH, "head_segx.png"))
        Image.fromarray(seg_head_hands_feet).save(
            os.path.join(DEBUG_OUTPUT_PATH, "head_hands_feet_segx.png")
        )
        Image.fromarray(seg_body_hands_feet).save(
            os.path.join(DEBUG_OUTPUT_PATH, "body_nohands_nofeet_segx.png")
        )

    mask_body = seg_body_hands_feet.astype(np.uint8)
    kernel_body = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (mask_padding + 1, mask_padding + 1)
    )
    mask_body = cv2.dilate(mask_body, kernel_body).astype(np.uint8) * 255

    if DEBUG:
        plt.imsave(
            os.path.join(DEBUG_OUTPUT_PATH, "circular_kernelx.png"),
            kernel_body,
            cmap="gray",
        )
        plt.imsave(
            os.path.join(DEBUG_OUTPUT_PATH, "mask_bodyx.png"), mask_body, cmap="gray"
        )

    mask_head = seg_head.astype(np.uint8)
    kernel_head = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (head_erosion * 2 + 1, head_erosion * 2 + 1)
    )
    mask_head = cv2.erode(mask_head, kernel_head).astype(np.uint8) * 255

    mask_head_hands_feet = seg_head_hands_feet.astype(np.uint8)
    kernel_sub = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (mask_erosion * 2 + 1, mask_erosion * 2 + 1)
    )
    mask_head_hands_feet = (
        cv2.erode(mask_head_hands_feet, kernel_sub).astype(np.uint8) * 255
    )

    mask_body_final = (
        np.logical_and(mask_body, np.logical_not(mask_head_hands_feet)).astype(np.uint8)
        * 255
    )
    if DEBUG:
        plt.imsave(os.path.join(DEBUG_OUTPUT_PATH, "maskx.png"), mask_body, cmap="gray")
        plt.imsave(
            os.path.join(DEBUG_OUTPUT_PATH, "mask_headx.png"), mask_head, cmap="gray"
        )
        plt.imsave(
            os.path.join(DEBUG_OUTPUT_PATH, "mask_head_hands_feetx.png"),
            mask_head_hands_feet,
            cmap="gray",
        )
        plt.imsave(
            os.path.join(DEBUG_OUTPUT_PATH, "mask_body_finalx.png"),
            mask_body_final,
            cmap="gray",
        )
    body = Image.fromarray(mask_body_final)
    head = Image.fromarray(mask_head)
    return body, head


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def preprocess(img, body, head, square_fn=square_pad, size=512):
    if isinstance(img, Image.Image):
        bg = img.getpixel((0, 0))
    elif isinstance(img, torch.Tensor):
        bg = img[0, 0]
    else:
        raise Exception(f"Unknown image type {type(bg)}")
    img_square = square_fn(img, bg).resize((size, size))
    seg_square = square_fn(body, (0,)).resize(
        (size, size), resample=Image.Resampling.NEAREST
    )
    head_square = square_fn(head, (0,)).resize(
        (size, size), resample=Image.Resampling.NEAREST
    )

    if DEBUG:
        img_square.save(os.path.join(DEBUG_OUTPUT_PATH, "img_squarex.jpg"))
        seg_square.save(os.path.join(DEBUG_OUTPUT_PATH, "seg_squarex.png"))

    return img_square, seg_square, head_square


def run_model(
    model,
    openpose_model,
    img,
    seg,
    prompt,
    guidance_scale=20,
    num_images_per_prompt=1,
    num_inference_steps=100,
) -> Union[List[Image.Image], np.ndarray]:
    posprompt = "photograph, beautiful, detailed, detailed shoes, photorealism, detailed hands, detailed feet, detailed fingers, realistic lighting, natural lighting, crystal clear, detailed skin, ultra focus, sharp quality, preserve gender"
    negprompt = "disfigured, ugly, bad, cartoon, anime, 3d, painting, bad hands, bad feet, deformed hands, broken anatomy, deformed, unrealistic, missing body parts, unclear, blurry"
    prompt = prompt + posprompt

    # use the pipeline to generate the image
    pose = openpose_model(img)
    pose.save("tmp/pose.png")
    control_image = make_inpaint_condition(img, seg)
    images = model(
        prompt=prompt,
        # image=img,
        image=img,
        # control_image=pose,
        control_image=[control_image, pose],
        mask_image=seg,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        negative_prompt=negprompt,
        num_inference_steps=num_inference_steps,
    ).images
    return images


def restore_head(img_orig: Image.Image, img_gen: Image.Image, head: Image.Image):
    if DEBUG:
        img_gen.save(os.path.join(DEBUG_OUTPUT_PATH, "pre_restorationx.jpg"))
    # finally, copy the head from the original image to fix potential distortions
    img_gen.paste(img_orig.convert("RGB"), (0, 0), mask=head)
    if DEBUG:
        head.save(os.path.join(DEBUG_OUTPUT_PATH, "head_maskx.png"))
        blank = Image.new(mode=img_gen.mode, size=img_gen.size)
        blank.paste(img_orig.convert("RGB"), (0, 0), mask=head)
        blank.save(os.path.join(DEBUG_OUTPUT_PATH, "head_imagex.jpg"))
        img_gen.save(os.path.join(DEBUG_OUTPUT_PATH, "restored_headx.jpg"))
    return img_gen


def postprocess(img: Image.Image, orig_size: Tuple[int, int]):
    scaled_size = [round(d / max(orig_size) * 512) for d in orig_size]

    square_size = img.size

    left = max((square_size[0] - min(scaled_size[0], 512)) // 2, 0)
    top = max((square_size[1] - min(scaled_size[1], 512)) // 2, 0)
    right = left + scaled_size[0]
    bottom = top + scaled_size[1]
    return img.crop((left, top, right, bottom))


def design_garment(
    model,
    openpose_model,
    img: Image.Image,
    seg: Image.Image,
    prompt,
    mask_dilation=70,
    mask_erosion=3,
    head_erosion=5,
    guidance_scale=20,
    num_images_per_prompt=1,
    num_inference_steps=200,
    square_fn=square_pad,
    size=512,
    do_postprocess=True,
):
    img_size = img.size
    if DEBUG:
        plt.imsave(os.path.join(DEBUG_OUTPUT_PATH, "/segmentation.png"), seg)

    body, head = generate_masks(
        seg,
        mask_padding=mask_dilation,
        mask_erosion=mask_erosion,
        head_erosion=head_erosion,
    )
    if DEBUG:
        blank = Image.new(mode=img.mode, size=img.size)
        blank.paste(img.convert("RGB"), (0, 0), mask=body)
        blank.save(os.path.join(DEBUG_OUTPUT_PATH, "/masked_bodyx.jpg"))
        blank2 = Image.new(mode=img.mode, size=img.size)
        blank2.paste(img.convert("RGB"), (0, 0), mask=ImageOps.invert(body))
        blank2.save(os.path.join(DEBUG_OUTPUT_PATH, "/preserve_areax.jpg"))

    img, body, head = preprocess(
        img, body, head, square_fn=square_fn, size=size
    )

    imgs = run_model(
        model,
        openpose_model,
        img,
        body,
        prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
    )
    imgs_final = [restore_head(img, img_gen, head) for img_gen in imgs]
    if do_postprocess:
        imgs_final = [postprocess(img_fin, img_size) for img_fin in imgs_final]
    return imgs_final


def main():
    # Model settings
    model_id = "runwayml/stable-diffusion-v1-5"

    # Set paths
    # images_dir = "sample_data/images"
    images_dir = "data/viton/images"
    # densepose_dir = "output/densepose"
    densepose_dir = "data/viton/densepose_dumps"
    output_dir = "output/openpose_control"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Model input
    image_id = "000031_0"
    # prompt = "Knee-length summer dress in blue with small pink flowers."
    prompt = "Bright yellow short-sleeve t-shirt."

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    # Load pipeline
    # controlnet = ControlNetModel.from_pretrained(
    #     "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
    # )
    controlnet = [
        ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16),
        ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
    ]
    
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_saftensors=True,
        safety_checker=None
    )
    pipeline = pipeline.to("cuda")

    # Load data
    image_path = os.path.join(images_dir, f"{image_id}.jpg")
    dp_path = os.path.join(densepose_dir, f"{image_id}.pkl")
    img, seg = load_data(image_path, dp_path)

    # Generate and save results
    imgs = design_garment(
        pipeline, openpose, img, seg, prompt, num_images_per_prompt=5
    )
    for i, img in enumerate(imgs):
        out_name = os.path.join(output_dir, f"{image_id}_yellowtshirt2_{i}.jpg")
        img.save(out_name)
        print(f"Saved generated image to: {out_name}")


if __name__ == "__main__":
    main()
