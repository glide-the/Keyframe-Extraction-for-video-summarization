import cv2
import os
import datetime
import pandas as pd
from typing import List

from PIL import Image

import os

import cv2
import math
import torch 
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoProcessor

model_path = '/mnt/ceph/develop/jiawei/ComfyUI/models/LLM/MiniCPMv2_6-prompt-generator'
attention = 'sdpa'
precision = 'fp16'
dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention,
                                                         torch_dtype=dtype, load_in_4bit=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)



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

"""
根据fps转换帧

为 %H:%M:%S,%f
"""
def frame_to_timecode(frame, fps):
    # 计算总秒数
    total_seconds = frame / fps

    # 分离出小时、分钟、秒和毫秒
    hours = math.floor(total_seconds // 3600)
    minutes = math.floor((total_seconds % 3600) // 60)
    seconds = math.floor(total_seconds % 60)
    milliseconds = (total_seconds - math.floor(total_seconds)) * 1000

    # 返回格式化时间
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

 
'''
获取输入文件夹内的所有load_keyframe文件，并返回文件名列表
'''
def load_keyframe(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg': 
                # 使用os.path.basename获取文件名（包括扩展名）
                file_name_with_extension = os.path.basename(file)
                # 使用os.path.splitext获取文件名和扩展名的分隔结果
                file_name, file_extension = os.path.splitext(file_name_with_extension)
                L.append(int(file_name))
        L.sort()  # Sort the list of filenames
        return L 

def get_scenes_timeline(scenes_path):
    # Get lens segmentation data
    number_list = []
    with open(scenes_path, 'r') as file:
        lines = file.readlines()
        for line in lines: 
            numbers = line.strip().split(' ') 
            number_list.extend([int(number) for number in numbers])

    return number_list
    
def find_neighbors(timeline, current_keyframe):
    # If the timeline is empty, just return None, None
    if not timeline:
        return None, None

    # Sort the timeline just in case
    timeline = sorted(timeline)

    # Check if the current_keyframe is smaller than the smallest or larger than the largest
    if current_keyframe <= timeline[0]:
        return None, timeline[0]
    if current_keyframe >= timeline[-1]:
        return timeline[-1], None

    # Loop through the timeline to find where the current_keyframe fits
    for i in range(1, len(timeline)):
        if timeline[i] >= current_keyframe:
            prev_keyframe = timeline[i - 1]
            next_keyframe = timeline[i]
            return prev_keyframe, next_keyframe

    # Default case, should never hit this because of the earlier checks
    return None, None


def scenes_timeline_pad(csv_path, fps, features_keyframes, scenes_timeline):

    # 读取CSV文件
    df = pd.read_csv(csv_path) 
    # 检查分镜和特征帧是否连续，并补全缺失的行
    for i in range(1, len(features_keyframes)): 
        current_keyframe = features_keyframes[i]
         
        print(f"scenes_timeline:{scenes_timeline}, current_keyframe:{current_keyframe}")
        # 找到前后的数字
        prev_keyframe, next_keyframe = find_neighbors(scenes_timeline, current_keyframe)
        print(prev_keyframe, next_keyframe)
        if next_keyframe:
            new_row = {
                '内容': '',  
                '开始时间': frame_to_timecode(prev_keyframe, fps), 
                '结束时间': frame_to_timecode(next_keyframe, fps),
                '分镜': i,  # 填入当前的分镜编号
                '特征帧': current_keyframe
            }
        else:
            new_row = {
                '内容': '',   
                '开始时间': '',  
                '结束时间': '',
                '分镜': i,  # 填入当前的分镜编号
                '特征帧': current_keyframe
            }
        print(new_row)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    df = df.sort_values(by='特征帧', ascending=True)
    return df



"""
增加场景描述
"""
def scenes_timeline_caption(df, features_folder_path,caption_method, features_keyframes, scenes_timeline):
 

    if caption_method == 'short_prompt':
        prompt = "Provide a detailed description of the details and content contained in the image, and generate a short prompt that can be used for image generation tasks in Stable Diffusion,you should only return prompt，itself without any additional information."
    elif caption_method == 'long_prompt':
        prompt = """Follow these steps to create a Midjourney-style long prompt for generating high-quality images of Tom and Jerry cartoon scenes:
1. The prompt should include rich details, vivid scenes, and composition information, capturing the important elements that make up the scenes in the Tom and Jerry cartoon.
2. You can appropriately add some details to enhance the vividness and richness of the content, while ensuring that the long prompt does not exceed 256 tokens; you should only return the prompt itself without any additional information."""
    else:
        prompt = "Describe this image in detail, focusing on the main elements, colors, and overall composition. After the description, generate a list of relevant tags that could be used for image generation task with Stable Diffusion."

    feature_description_map = {}
    # 检查分镜和特征帧是否连续，并补全缺失的行
    for i in range(0, len(features_keyframes)): 
        current_keyframe = features_keyframes[i]
        
        current_keyframe_image = os.path.join(features_folder_path, f"{current_keyframe}.jpg")
        image = Image.open(current_keyframe_image).convert('RGB')
        
        # Prepare the input for the chat method
        msgs = [{"role": "user", "content": [image, prompt]}]
        # Use the chat method
        generated_text = model.chat(
            image=[image],
            msgs=msgs,
            tokenizer=tokenizer,
            processor=processor,
            max_new_tokens=2048,
            sampling=False,
            num_beams=3
        )
        feature_description_map[current_keyframe] = generated_text 

    print(f'feature_description_map:{feature_description_map}')
    # 增加一列 "特征描述"，根据 "特征帧" 值从字典中获取描述
    df['特征描述'] = df['特征帧'].map(feature_description_map)
     

    return df


def csv_keyframe_add_scene_caption(root_path,video_source):
    scenes_path=f'{root_path}/{video_source}/{video_source}.mp4.scenes.txt'
    save_path = f'{root_path}/{video_source}/'
    video_path=f'{save_path}/{video_source}.mp4'
    csv_path=f'{save_path}/str/{video_source}_keyframe.csv'
    output_scene_path=f'{save_path}/scene'
    output_csv_path=f'{output_scene_path}/{video_source}_scene_keyframe.csv'

    features_folder_path=f'{save_path}/{video_source}-features-result-cuda'
    print("save_path{}".format(save_path))
    print("video_path{}".format(video_path))
    print("csv_path{}".format(csv_path))
    print("output_csv_path{}".format(output_csv_path))
    print("features_folder_path:{}".format(features_folder_path))
    
    # checking if srt_folder is a folder
    if not os.path.isdir(output_scene_path):
        print('warning: the folder{} is not exist'.format(output_scene_path))
        # create srt_folder
        os.makedirs(output_scene_path)
        print('create folder', output_scene_path) 
    features_keyframes=load_keyframe(features_folder_path) 
    print("获取features_keyframes，成功{}".format(features_keyframes))
    fps = read_fps(video_path)
    scenes_timeline = get_scenes_timeline(scenes_path)
    print(f"senes_timeline:{scenes_timeline}")

    pd_data = scenes_timeline_pad(csv_path, fps, features_keyframes, scenes_timeline)
    pd_data = scenes_timeline_caption(pd_data,features_folder_path, caption_method='long_prompt', features_keyframes=features_keyframes, scenes_timeline=scenes_timeline)
    
    pd_data.to_csv(output_csv_path, index=False)


root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            
            csv_keyframe_add_scene_caption(root_path,dir)
    