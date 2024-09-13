import cv2

import os

import pandas as pd
from typing import List
import datetime

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




def is_frame_in_time_interval(frame_number: int, start_time:str, end_time:str, fps:int):
    # 将开始时间和结束时间解析为时间对象
    start_time_obj = datetime.datetime.strptime(start_time, "%H:%M:%S,%f")
    end_time_obj = datetime.datetime.strptime(end_time, "%H:%M:%S,%f")

    # 计算时间区间的秒数
    start_seconds = (start_time_obj.hour * 3600 + start_time_obj.minute * 60 + start_time_obj.second +
                     start_time_obj.microsecond / 1000000)
    end_seconds = (end_time_obj.hour * 3600 + end_time_obj.minute * 60 + end_time_obj.second +
                   end_time_obj.microsecond / 1000000)

    # 计算帧在时间区间内的时间
    frame_time = frame_number // fps
    print(frame_time)
    print(start_seconds)
    # 检查帧时间是否在时间区间内
    return start_seconds <= frame_time <= end_seconds
 

def add_scene_column(input_csv_file, output_csv_file, fps, keyframes: List[int] = []):
    # 读取CSV文件
    df = pd.read_csv(input_csv_file)

    # 初始化分镜列
    df['分镜'] = 0
    pre_k_index = 0
    pre_k_frame_number = 0
    
    # 遍历每一行
    for index, row in df.iterrows():
        start_time = row['开始时间']
        end_time = row['结束时间']
        _flag = False
        for k_index, frame_number in enumerate(keyframes): 
            is_frame = is_frame_in_time_interval(frame_number=frame_number,
                                                 start_time=start_time, 
                                                 end_time=end_time,
                                                 fps=fps)
            if is_frame:
                # 判断是否应增加分镜值
                df.at[index, '分镜'] = k_index
                df.at[index, '特征帧'] = int(frame_number)
                pre_k_index = k_index
                pre_k_frame_number = int(frame_number)
                _flag=True

        # 如果不存在分镜片段，获取上次的序号
        if not _flag:
            df.at[index, '分镜'] = pre_k_index
            df.at[index, '特征帧'] = frame_number
                
    df = df.sort_values(by='特征帧', ascending=True)

    # 保存带有新列的CSV文件
    df.to_csv(output_csv_file, index=False)

 
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


def csv_keyframe_merge(root_path,video_source):
    save_path = f'{root_path}/{video_source}/'
    video_path=f'{save_path}/{video_source}.mp4'
    csv_path=f'{save_path}/str/{video_source}.csv'
    output_csv_path=f'{save_path}/str/{video_source}_keyframe.csv'
    
    features_folder_path=f'{save_path}/{video_source}-features-result-cuda'
    print("save_path{}".format(save_path))
    print("video_path{}".format(video_path))
    print("csv_path{}".format(csv_path))
    print("output_csv_path{}".format(output_csv_path))
    print("features_folder_path:{}".format(features_folder_path))
 
    keyframes=load_keyframe(features_folder_path) 
    print("获取keyframes，成功{}".format(keyframes))
    fps = read_fps(video_path)
    add_scene_column(input_csv_file=csv_path, output_csv_file=output_csv_path, fps=fps, keyframes=keyframes)
    

root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            
            csv_keyframe_merge(root_path,dir)
    