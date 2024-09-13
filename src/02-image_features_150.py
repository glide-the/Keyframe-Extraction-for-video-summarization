# require modelscope>=0.3.7，目前默认已经超过，您检查一下即可
# 按照更新镜像的方法处理或者下面的方法
# pip install --upgrade modelscope -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
# 需要单独安装decord，安装方法：pip install decord
 
import os

import pickle
from tqdm import tqdm  # Import the tqdm function or class
import numpy as np
import torch
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.image import load_image

pipeline = pipeline(task=Tasks.multi_modal_embedding,
    model='/mnt/ceph/develop/jiawei/lora_dataset/checkpoint/multi-modal_clip-vit-large-patch14_336_zh/damo/multi-modal_clip-vit-large-patch14_336_zh', model_revision='v1.0.1')
 

'''
获取输入文件夹内的所有jpg文件，并返回文件名全称列表
'''
def load_jpg(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                filename=os.path.join(root, file)
                L.append(filename)
        return L 
        
def batch(iterable, size):
    # range对象的step是size
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]




def batch_embedding_img(data_folder, jpg_files):
    #对每一个文件进行操作
    batch_len=3
    batch_list = batch(jpg_files,batch_len)
    b_unit = tqdm(enumerate(jpg_files),total=len(jpg_files))
    
    img_embedding_list = []
    
    for batch_file in batch_list:  
        table_data=[]
        for filename in batch_file:
            
            input_img = load_image(filename)
            # 支持一张图片(PIL.Image)或多张图片(List[PIL.Image])输入，输出归一化特征向量
            # 2D Tensor, [图片数, 特征维度]
            img_embedding = pipeline.forward({'img': input_img})['img_embedding']
            #torch.Size([1, 768])形状转换 torch.Size([768])
            tensor_squeezed = img_embedding.squeeze()
            img_embedding_list.append(tensor_squeezed)
        # 更新进度
        b_unit.update(batch_len) 
    
    
    # 循环结束后关闭进度条
    b_unit.close()
    
    save_name = os.path.join(data_folder, f"features.pkl")
    # Open the file in write mode
    with open(save_name, 'wb') as file:
        
        features = [tensor.cpu() for tensor in img_embedding_list]
        features_np = [tensor.numpy() for tensor in features] 
        print(np.asarray(features_np).shape)
        pickle.dump(features_np, file)
        file.close
     
    
    
    # Read inference data from local
    with open(save_name, 'rb') as file:
        features = pickle.load(file)
    
    
    features = np.asarray(features)
    
    print(features.shape)
      
root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠150/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs: 
            data_folder=f'{root_path}/{dir}/keyframe-features' 
            
            jpg_files=load_jpg(data_folder) 
            print(f"获取数据{data_folder}，成功{len(jpg_files)}")
            batch_embedding_img(data_folder,jpg_files)
    
