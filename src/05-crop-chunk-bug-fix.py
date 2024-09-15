import os
 

def modify_video_index(videos_folder):
     
    for video_file in os.listdir(videos_folder):
        # 提取文件名和扩展名
        file_name, file_extension = os.path.splitext(video_file)
        
        # 提取索引并增加1
        parts = file_name.split('_')
        base_name = '_'.join(parts[:-1])
        index = int(parts[-1]) - 1
        
        # 生成新的文件名
        new_file_name = f"{base_name}_{index}{file_extension}"
        
        # 重命名文件
        old_path = os.path.join(videos_folder, video_file)
        new_path = os.path.join(videos_folder, new_file_name)
        os.rename(old_path, new_path)
        
        print(f"Renamed {video_file} to {new_file_name}")
        

root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠6/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            videos_folder = f'{root_path}/{dir}/scene_chunks/videos'
            modify_video_index(videos_folder)
