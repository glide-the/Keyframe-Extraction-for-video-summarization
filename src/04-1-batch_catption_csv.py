
import sys
sys.path.append('/mnt/ceph/develop/jiawei/lora_dataset/Chat-Haruhi-Suzumiya')


from yuki_builder.srt2csv import convert 

import os
import pathlib

from yuki_builder.run_whisper import Video2Subtitles



v2s = Video2Subtitles()

def to_str(root_path, dir):
    input_file = f'{root_path}/{dir}/{dir}.mp4'
    srt_folder = f'{root_path}/{dir}/str'
    print('input_file:'+input_file)
    print('srt_folder:'+srt_folder)
    result = v2s.transcribe(input_file, srt_folder)
    print(result)


import os

root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            to_str(root_path,dir)



def srt2csv(input_srt,output_folder):
    # checking if srt_folder is a folder
    if not os.path.isdir(output_folder):
        print('warning: the folder{} is not exist'.format(output_folder))
        # create srt_folder
        os.makedirs(output_folder)
        print('create folder', output_folder)
     
    if not os.path.isfile(input_srt):
        print('Error: The input file {} is not exist'.format(input_srt))
        return
    
    # checking if input_srt is a srt_file
    if not (pathlib.Path(input_srt).suffix == '.srt' or pathlib.Path(input_srt).suffix == '.ass'):
        print('Error: The input file {} must be a .srt or .ass file'.format(input_srt))
        return
    convert(input_srt, output_folder, True)


def to_srt2csv(root_path, dir):
    input_file = f'{root_path}/{dir}/str/{dir}.srt'
    srt_folder = f'{root_path}/{dir}/str'
    print('input_file:'+input_file)
    print('srt_folder:'+srt_folder)
    srt2csv(input_file, srt_folder)



root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            to_srt2csv(root_path,dir)
    