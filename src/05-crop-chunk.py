import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import cv2
import pandas as pd
import os
import math
import datetime
import torch
import numpy as np
from torch.nn import functional as F
from typing import Union, List
from diffusers.image_processor import VaeImageProcessor

import PIL.Image
from diffusers.utils import export_to_video


import os

def read_fps(video_path:str):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    
    # 获取视频的帧速率
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
     
    # 释放视频捕获对象
    video_capture.release()
    
    # 关闭视频文件
    cv2.destroyAllWindows()
    return fps

def pad_frame(img, scale):
    _, _, h, w = img.shape
    tmp = max(32, int(32 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0,  pw - w, 0, ph - h)
    return F.pad(img, padding)

def pad_video_of_np(samples, w, h):
    print(f"samples dtype:{samples.dtype}")
    print(f"samples shape:{samples.shape}")
    output = []
    # [f, c, h, w]
    for b in range(samples.shape[0]):
        frame = samples[b : b + 1]
        frame = pad_frame(frame, 1).to(dtype=samples.dtype)
        
        frame = F.interpolate(frame, size=(h, w))
        output.append(frame.squeeze(0)) # (to [f, w, h, c])

    image_np = VaeImageProcessor.pt_to_numpy(torch.stack(output))  # (to [49, 512, 480, 3])
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
    return image_pil


def split_video(video_path, output_dir, start_time, end_time, features_frame, w=540 , h=360 ,chunk_duration=4):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Get the frames per second
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video
    output_files = []
    # Calculate start and end frames based on time
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set video position to start frame

    current_frame = start_frame
    chunk_index = 0

    while current_frame <= end_frame:
        chunk_frames = []
        for _ in range(int(chunk_duration * fps)):
            ret, frame = video_capture.read()
            if not ret:
                break  
            frame_rgb = frame[..., ::-1]
            
            # 创建一个新数组，确保 stride 是正的
            frame_rgb = frame_rgb.copy()
            tensor = torch.from_numpy(frame_rgb).float().to("cpu", non_blocking=True).float() / 255.0
            chunk_frames.append(
                tensor.permute(2, 0, 1)
            )  # to [c, h, w,]
            current_frame += 1

        if len(chunk_frames) == 0:
            break
            
        # If the last chunk is shorter than expected, repeat the last frame
        while len(chunk_frames) < int(chunk_duration * fps):
            chunk_frames.append(chunk_frames[-1])
            
        pt_frame = torch.from_numpy(np.stack(chunk_frames))  # to [f, c, h, w]
         
        chunk_images = pad_video_of_np(pt_frame,w,h)
     
        # Save the chunk as a video
        output_file = os.path.join(output_dir, f'chunk_{features_frame}_{chunk_index}.mp4')
        print(output_file)
        save_chunk_as_video(chunk_images, output_file, fps=math.ceil((len(chunk_images) - 1) / 6))
        output_files.append(output_file)
        chunk_index += 1

    video_capture.release()
    return output_files
    
 
    
def save_chunk_as_video(tensor: Union[List[np.ndarray], List[PIL.Image.Image]],video_path, fps: int = 8):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path, fps=fps)
    return video_path


 

def loadcsv_scene_chunk(scene_csv_path, video_path, videos_output_dir, labels_output_dir):
    df = pd.read_csv(scene_csv_path)
    
    # 计算每个分镜编号的总持续时间
    df['持续时间'] = (pd.to_datetime(df['结束时间']) - pd.to_datetime(df['开始时间'])).dt.total_seconds()
    grouped_df = df.groupby('分镜')['持续时间'].sum().reset_index()
    df = df[df['持续时间'] >= 3]
    
    df = df.sort_values(by='持续时间', ascending=True)
    # Iterate over each row in the table
    for _, row in df.iterrows():
        start_time = row['开始时间']
        end_time = row['结束时间']
        features_caption = row['特征描述']
        features_frame = int(row['特征帧'])
        # 将开始时间和结束时间解析为时间对象
        start_time_obj = datetime.datetime.strptime(start_time, "%H:%M:%S,%f")
        end_time_obj = datetime.datetime.strptime(end_time, "%H:%M:%S,%f")
    
        # 计算时间区间的秒数
        start_seconds = (start_time_obj.hour * 3600 + start_time_obj.minute * 60 + start_time_obj.second +
                         start_time_obj.microsecond / 1000000)
        end_seconds = (end_time_obj.hour * 3600 + end_time_obj.minute * 60 + end_time_obj.second +
                       end_time_obj.microsecond / 1000000)
        start_seconds = start_seconds-0.5
        end_seconds = end_seconds-0.5
        # Split the video for each start and end time
        output_files = split_video(video_path, videos_output_dir, start_seconds, end_seconds,features_frame)
        for index, path in enumerate(output_files):
            
            lable_file = os.path.join(labels_output_dir, f'chunk_{features_frame}_{index}.txt')
            # Open the file in write mode
            with open(lable_file, 'w') as file: 
                file.write(features_caption)
              


def check_output_folder(output_folder):
    if not os.path.isdir(output_folder):
        print('warning: the folder{} is not exist'.format(output_folder))
        # create srt_folder
        os.makedirs(output_folder)
        print('create folder', output_folder)

def split_video_with_chunk(root_path, video_source):
    save_path = f'{root_path}/{video_source}/'
    video_path=f'{save_path}/{video_source}.mp4' 
 
    scene_csv_path=f'{save_path}/scene/{video_source}_scene_keyframe.csv'
    videos_output_dir = f'{save_path}/scene_chunks/videos'
    labels_output_dir = f'{save_path}/scene_chunks/labels'

    check_output_folder(videos_output_dir)

    check_output_folder(labels_output_dir)
    
    loadcsv_scene_chunk(scene_csv_path,video_path,videos_output_dir, labels_output_dir)
    



root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            split_video_with_chunk(root_path,dir)
    
