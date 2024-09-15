import sys
from rife_model import load_rife_model, rife_inference_with_latents
from huggingface_hub import hf_hub_download, snapshot_download
import torch
import os
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video
import math

import os
import shutil
import cv2
import numpy as np
import utils
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

snapshot_download(repo_id="AlexWortega/RIFE", local_dir="model_rife")


torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

frame_interpolation_model = load_rife_model("model_rife")

frame_interpolation_model.device()


def loadvideo_pt(video_path):
    
    video_capture = cv2.VideoCapture(video_path)
    tot_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames
    pt_frame_data = []
    
    # Read each frame using VideoCapture
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        # BGR to RGB
        frame_rgb = frame[..., ::-1]
        frame_rgb = frame_rgb.copy()
        tensor = torch.from_numpy(frame_rgb).float().to("cpu", non_blocking=True).float() / 255.0
        pt_frame_data.append(
            tensor.permute(2, 0, 1)
        )  # to [c, h, w,]
    
    video_capture.release()  # Release the video after done
    
    pt_frame = torch.from_numpy(np.stack(pt_frame_data))
    pt_frame = pt_frame.to(device)
    pt_frame = pt_frame.unsqueeze(0)
    return pt_frame.to(dtype=torch.float16)



def rile_with_path(video_path, export_video_path):
    pt_frame = loadvideo_pt(video_path)
    rife_pt_frame = rife_inference_with_latents(frame_interpolation_model,pt_frame )
    
    pt_image = rife_pt_frame[0]
    pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])
     
    image_np = VaeImageProcessor.pt_to_numpy(pt_image)
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
     
    export_to_video(image_pil, export_video_path, fps=math.ceil((len(image_pil) - 1) / 6))




def load_file_by_extension(file_dir, extension):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == f'.{extension}': 
                L.append(file) 
        return L 

def rife_video_with_chunk(root_path, video_source):
    save_path = f'{root_path}/{video_source}/' 
 
    scene_csv_path=f'{save_path}/scene/{video_source}_scene_keyframe.csv'
    videos_output_dir = f'{save_path}/scene_chunks/videos'
    labels_output_dir = f'{save_path}/scene_chunks/labels'
    videos_files = load_file_by_extension(videos_output_dir, 'mp4')
    labels_files = load_file_by_extension(labels_output_dir, 'txt')

    print(f"load videos_files:{videos_files}")
    for i, filename in enumerate(videos_files):
        video_path = f'{videos_output_dir}/{filename}'
        export_video_path = f'{videos_output_dir}/rife_{filename}'
        rile_with_path(video_path,export_video_path)
    

    for i, lable_filename in enumerate(labels_files):
        lable_path = f'{videos_output_dir}/{lable_filename}'
        export_lable_path = f'{videos_output_dir}/rife_{lable_filename}'
        shutil.copy(lable_path, export_lable_path)

    

root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠2/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            rife_video_with_chunk(root_path,dir)
    
