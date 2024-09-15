import shutil
import os

def load_file_by_extension(file_dir, extension):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == f'.{extension}': 
                L.append(file) 
        return L 

def check_output_folder(output_folder):
    if not os.path.isdir(output_folder):
        print('warning: the folder{} is not exist'.format(output_folder))
        # create srt_folder
        os.makedirs(output_folder)
        print('create folder', output_folder)
        
def dataset_copy(root_path, video_source, dir_put):
    save_path = f'{root_path}/{video_source}/' 
    videos_source_dir  = f'{save_path}/scene_chunks/videos'
    labels_source_dir  = f'{save_path}/scene_chunks/labels'
    if not os.path.exists(videos_source_dir):
        print( f"视频源文件夹不存在: {videos_source_dir}")
        return
    if not os.path.exists(labels_source_dir):
        print( f"视频源文件夹不存在: {labels_source_dir}")
        return

    
    videos_files = load_file_by_extension(videos_source_dir, 'mp4')
    labels_files = load_file_by_extension(labels_source_dir, 'txt')

    # 定义目标文件夹路径
    videos_output_dir = os.path.join(dir_put, 'videos')
    labels_output_dir = os.path.join(dir_put, 'labels')
    
    check_output_folder(videos_output_dir)
    check_output_folder(labels_output_dir)

    for i, filename in enumerate(videos_files):
     
        # Copy video to output directory
        video_path = f'{videos_source_dir}/{filename}' 
        output_video_path = f'{videos_output_dir}/{video_source}_{filename}' 
        shutil.copy(video_path, output_video_path)
         
    for i, filename in enumerate(labels_files):
     
        # Copy video to output directory
        label_path = f'{labels_source_dir}/{filename}' 
        output_lable_path = f'{labels_output_dir}/{video_source}_{filename}' 
        shutil.copy(label_path, output_lable_path)

 

dir_put = '/mnt/ceph/develop/jiawei/lora_dataset/cogvideox_sat_tom_jerry_lora_dataset'
root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠150/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            dataset_copy(root_path,dir,dir_put)
    

 