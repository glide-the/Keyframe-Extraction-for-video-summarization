from extraction.Keyframe_extraction import scen_keyframe_extraction

import os

def k_means_keyframe(root_path,video_source):
    scenes_path=f'{root_path}/{video_source}/{video_source}.mp4.scenes.txt'
    features_path=f'{root_path}/{video_source}/keyframe-features/features.pkl'
    video_path=f'{root_path}/{video_source}/{video_source}.mp4'
    save_path = f'{root_path}/{video_source}/'
    folder_path=f'{video_source}-features-result-cuda'
    print(f"scenes_path:{scenes_path}")
    print(f"features_path:{features_path}")
    print(f"video_path:{video_path}")
    print(f"save_path:{save_path}")
    print(f"folder_path:{folder_path}") 
    
    
    scen_keyframe_extraction(scenes_path=scenes_path,
                            features_path=features_path,
                            video_path=video_path,
                            save_path=save_path,
                            folder_path=folder_path)


root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            k_means_keyframe(root_path,dir)
    