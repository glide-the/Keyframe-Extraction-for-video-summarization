from scripts.save_keyframe import save_frames

import os

def import_save_keyframe(scenes_file, video_path, save_path):
    # 打开文件并读取内容
    with open(scenes_file, 'r') as file:
        lines = file.readlines()
    
    # 初始化一个空的Python数组
    result = []
    
    # 循环遍历文件中的每一行
    for line in lines:
        # 分割每行的内容，并将其转换为整数
        start, end = map(int, line.split())
        
        # 创建一个包含开始帧和结束帧的元组，并添加到结果数组中
        result.append((start, end))
    
    # 打印结果数组
    print(result)
    keyframe_indexes = [item[0] for item in result]
    folder_name='keyframe'
    save_frames(keyframe_indexes=keyframe_indexes,video_path=video_path,save_path=save_path,folder_name=folder_name)


    ## 保存所有帧
    keyframe_all_indexes = []
    
    for start, end in result:
        keyframe_all_indexes.extend(range(start, end + 1))
    
    
    
    folder_name='keyframe-features'
    save_frames(keyframe_indexes=keyframe_all_indexes,video_path=video_path,save_path=save_path,folder_name=folder_name)




root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            scenes_file=f'{root_path}/{dir}/{dir}.mp4.scenes.txt'
            save_path=f'{root_path}/{dir}'
            video_path=f'{root_path}/{dir}/{dir}.mp4'
                        
            import_save_keyframe(scenes_file,video_path, save_path)
    