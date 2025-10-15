import os
import argparse
import imageio
from PIL import Image
from stylesculptor.pipelines import TrellisImageTo3DPipeline
from stylesculptor.utils import render_utils, postprocessing_utils
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--cnt', type=str, required=True, help='Content image paths')
parser.add_argument('--sty', type=str, required=True, help='Style image paths')
parser.add_argument('--sty_edge', type=str, required=True, help='Style edge image paths')
parser.add_argument('--intensity', type=int, default=3, help='Style intensity values (0-5)')
args = parser.parse_args()

def load_images(path):
    if os.path.isdir(path):
        return [Image.open(os.path.join(path, img)) for img in os.listdir(path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    elif os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return [Image.open(path)]
    else:
        raise ValueError(f"Invalid image path: {path}")

image_content = load_images(args.cnt)
image_style = load_images(args.sty)
image_style_edge = load_images(args.sty_edge)

content_name = os.path.splitext(os.path.basename(args.cnt))[0]
style_name = os.path.splitext(os.path.basename(args.sty))[0]
edge_name = os.path.splitext(os.path.basename(args.sty_edge))[0]
print(f"Content image: {content_name}, Style image: {style_name}, Edge image: {edge_name}")

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

gamma = 1.1  # Increase brightness
inv_gamma = 1.0 / gamma
table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
table = np.array(table, np.uint8)

for i in range(len(image_style)):
    image_style[i] = Image.fromarray(cv2.LUT(np.array(image_style[i]), table))

outputs = pipeline.run_multi_image(
    image_content,
    image_style,
    image_style_edge,
    intensity=args.intensity,
    seed=0,
    sparse_structure_sampler_params={
        "steps": 100,
        "cfg_strength": 6.5,
    },
    slat_sampler_params={
        "steps": 100,
        "cfg_strength": 3.5,
    },
)

video = render_utils.render_video(outputs['gaussian'][0])['color']
if not os.path.exists("./results"):
    os.makedirs("./results")
imageio.mimsave(f"./results/{content_name}_{style_name}_intensity{args.intensity}.mp4", video, fps=30)
