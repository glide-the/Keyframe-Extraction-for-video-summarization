import shutil
import os

def check_output_folder(output_folder):
    if not os.path.isdir(output_folder):
        print('warning: the folder{} is not exist'.format(output_folder))
        # create srt_folder
        os.makedirs(output_folder)
        print('create folder', output_folder)

def load_file_by_extension(file_dir, extension):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == f'.{extension}':  
                L.append(file) 
        return L 

def video_dataset_sat_to_hf(root_path, video_source, output_dir):
    save_path = f'{root_path}/{video_source}/' 
  
    videos_output_dir = f'{save_path}/scene_chunks/videos'
    labels_output_dir = f'{save_path}/scene_chunks/labels'
    videos_files = load_file_by_extension(videos_output_dir, 'mp4')
    labels_files = load_file_by_extension(labels_output_dir, 'txt')

    print(f"load videos_files: {len(videos_files)}")

    prompts_file = f'{output_dir}/captions.txt'
    videos_file = f'{output_dir}/videos.txt'

    instance_videos = []
    instance_prompts = []
    videos_dir = f'{output_dir}/videos'
    check_output_folder(videos_dir)
    for i, filename in enumerate(videos_files):
        video_path = f'{videos_output_dir}/{filename}' 
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = f'{labels_output_dir}/{label_filename}'

        if os.path.exists(video_path) and os.path.exists(label_path):
            # Copy video to output directory
            output_video_path = f'{videos_dir}/{video_source}_{filename}' 
            shutil.copy(video_path, output_video_path)
            instance_videos.append(f'videos/{video_source}_{filename}')

            # Read and append prompt
            with open(label_path, "r", encoding="utf-8") as file:
                prompt_data = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
                instance_prompts.append("".join(prompt_data))

    # Write instance_videos to videos_file
    with open(videos_file, "a", encoding="utf-8") as vf:
        for video in instance_videos:
            vf.write(video + '\n')

    # Write instance_prompts to prompts_file
    with open(prompts_file, "a", encoding="utf-8") as pf:
        for prompt in instance_prompts:
            pf.write(prompt + '\n')

    print(f"Copied {len(instance_videos)} videos and saved corresponding prompts.")


import os


dir_put = '/mnt/ceph/develop/jiawei/lora_dataset/hf_cogvideo_tom_jerry_dataset'
root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠/'
check_output_folder(dir_put)
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            video_dataset_sat_to_hf(root_path,dir,dir_put)

