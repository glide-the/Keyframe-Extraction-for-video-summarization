import cv2
import os
import datetime

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


def split_video(video_path, output_dir, start_time, end_time, features_frame, w=540 , h=360 ,chunk_duration=3):
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


from PIL import Image
from openai import OpenAI

client = OpenAI(
    # api_key="YOUR_API_KEY",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
# Function to encode the image
def encode_image(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def caption_chunk_gpt(video_path, features_caption):

    video_capture = cv2.VideoCapture(video_path)
    ret, first_frame = video_capture.read()
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, last_frame = video_capture.read()

    # Get the frame rate (frames per second)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Calculate the duration of the video in seconds
    video_duration = total_frames / fps
    
    video_capture.release()
    
    first_frame_rgb = first_frame[..., ::-1] 
    first_frame_rgb = first_frame_rgb.copy()

    last_frame_rgb = last_frame[..., ::-1] 
    last_frame_rgb = last_frame_rgb.copy()
    
    first_frame_pil_image = Image.fromarray(first_frame_rgb)

    last_frame_pil_image = Image.fromarray(last_frame_rgb)

    prompt = """Follow these steps to create a Midjourney-style long prompt for generating high-quality images: 
            1. The prompt should include rich details, vivid scenes, and composition information, capturing the important elements that make up the scene. 
            2. You can appropriately add some details to enhance the vividness and richness of the content, while ensuring that the long prompt does not exceed 256 tokens,you should only return prompt，itself without any additional information"""

    # Prepare the input for the chat method
    msgs = [{"role": "user", "content": [first_frame_pil_image, prompt]}]
    # Use the chat method
    first_frame_generated_text = model.chat(
        image=[first_frame_pil_image],
        msgs=msgs,
        tokenizer=tokenizer,
        processor=processor,
        max_new_tokens=2048,
        sampling=False,
        num_beams=3
    ) 
    
    # Prepare the input for the chat method
    msgs = [{"role": "user", "content": [last_frame_pil_image, prompt]}]
    # Use the chat method
    last_frame_generated_text = model.chat(
        image=[last_frame_pil_image],
        msgs=msgs,
        tokenizer=tokenizer,
        processor=processor,
        max_new_tokens=2048,
        sampling=False,
        num_beams=3
    ) 
    # Sample dictionary of image captions (this should be generated based on your video analysis)
    image_captions = { 
        "1": first_frame_generated_text,
        video_duration: last_frame_generated_text
    }
    
    # Convert image_captions dictionary into a format suitable for new_captions
    new_captions = "\n".join([f"{time}: '{description}'" for time, description in image_captions.items()])

    caption_summary_prompt = f"""Video description: \"{features_caption}\"\n\n
We extracted several frames from this video and described
each frame using an image understanding model, stored in the dictionary variable ‘image_captions: Dict[str: str]‘.
In ‘image_captions‘, the key is the second at which the image appears in the video, and the value is a detailed description
of the image at that moment. Please describe the content of this video in as much detail as possible, based on the
information provided by ‘image_captions‘, including the objects, scenery, animals, characters, and camera
movements within the video. \n image_captions={new_captions}\n\n
You should output your summary directly, and not mention
variables like ‘image_captions‘ in your response.
Do not include ‘\\n’ and the word ’video’ in your response.
Do not use introductory phrases such as: \"The video presents\", \"The video depicts\", \"This video showcases\",
\"The video captures\" and so on.\n Please start the description with the video content directly, such as \"A man
first sits in a chair, then stands up and walks to the kitchen....\"\n Do not use phrases like: \"as the video
progressed\" and \"Throughout the video\".\n Please describe  the content of the video and the changes that occur, in
chronological order.\n Please keep the description of this video within 100 English words."""

    print(f"caption_summary_prompt:{caption_summary_prompt}")
    # 减少信息，生成速度可以在0.1秒完成
    tools = [
        {
            "type": "web_search",
            "web_search": {
                "enable": False, 
            }
        }
    ]
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[ {"role": "user", "content": f"{caption_summary_prompt}"}],
        temperature=0,
        tools=tools,
        max_tokens=2000,
    ) 

    caption_summary_text = response.choices[0].message.content
    print(f"{video_path}\r\ncaption_summary_text:{caption_summary_text}")
    return first_frame_generated_text, last_frame_generated_text, caption_summary_text




def loadcsv_scene_chunk(scene_csv_path, video_path, videos_output_dir, labels_output_dir):
    df = pd.read_csv(scene_csv_path)
    
    # 计算每个分镜编号的总持续时间
    df['持续时间'] = (pd.to_datetime(df['结束时间']) - pd.to_datetime(df['开始时间'])).dt.total_seconds()
    grouped_df = df.groupby('分镜')['持续时间'].sum().reset_index()
    df = df[df['持续时间'] >= 3]
    
    df = df.sort_values(by='持续时间', ascending=True)
        
    # 创建一个新的 DataFrame 来存储行
    new_df = pd.DataFrame(columns=df.columns)
    # 为新数据添加列
    new_df['chunk_index'] = None
    new_df['chunk_path'] = None
    new_df['first_frame_generated_text'] = None
    new_df['last_frame_generated_text'] = None
    new_df['caption_summary_text'] = None
    # Iterate over each row in the table
    for _, row in df.iterrows():
        start_time = row['开始时间']
        end_time = row['结束时间']
        features_caption = row['特征描述']
        features_frame = int(row['特征帧'])
        # 将行转换为 DataFrame 并追加到 new_df
        new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)

        # 创建一个新的 DataFrame 来存储冗余行
        expanded_df = pd.DataFrame(columns=new_df.columns)
        # 将开始时间和结束时间解析为时间对象
        start_time_obj = datetime.datetime.strptime(start_time, "%H:%M:%S,%f")
        end_time_obj = datetime.datetime.strptime(end_time, "%H:%M:%S,%f")
    
        # 计算时间区间的秒数
        start_seconds = (start_time_obj.hour * 3600 + start_time_obj.minute * 60 + start_time_obj.second +
                         start_time_obj.microsecond / 1000000)
        end_seconds = (end_time_obj.hour * 3600 + end_time_obj.minute * 60 + end_time_obj.second +
                       end_time_obj.microsecond / 1000000)
        start_seconds = start_seconds-1
        end_seconds = end_seconds-1
        # Split the video for each start and end time
        output_files = split_video(video_path, videos_output_dir, start_seconds, end_seconds,features_frame)
        for index, path in enumerate(output_files):
            first_frame_generated_text, last_frame_generated_text, caption_summary_text = caption_chunk_gpt(path, features_caption)
            
            # Update the corresponding row in new_df with the generated text values
            new_df.at[new_df.index[-1], 'chunk_index'] = f'chunk_{features_frame}_{index}'
            new_df.at[new_df.index[-1], 'chunk_path'] = path
            new_df.at[new_df.index[-1], 'first_frame_generated_text'] = first_frame_generated_text
            new_df.at[new_df.index[-1], 'last_frame_generated_text'] = last_frame_generated_text
            new_df.at[new_df.index[-1], 'caption_summary_text'] = caption_summary_text
            
            lable_file = os.path.join(labels_output_dir, f'chunk_{features_frame}_{index}.txt')
            # Open the file in write mode
            with open(lable_file, 'w') as file: 
                file.write(caption_summary_text)

    return new_df




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

    scene_chunks_csv_path = f'{save_path}/scene_chunks/{video_source}_scene_chunk_frame.csv'
    check_output_folder(videos_output_dir)

    check_output_folder(labels_output_dir)
    
    new_df = loadcsv_scene_chunk(scene_csv_path,video_path,videos_output_dir, labels_output_dir)
    
    new_df.to_csv(scene_chunks_csv_path, index=False)


import os

root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠150/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            split_video_with_chunk(root_path,dir)
    