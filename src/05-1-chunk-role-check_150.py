 
import base64
import pandas as pd
from PIL import Image
from openai import OpenAI

client = OpenAI(
    # api_key=".wRDvUO0r3SqRTbrS",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
 


def caption_chunk_gpt_compare(video_path, caption_summary_text):
   
    with open(video_path, 'rb') as video_file:
        video_base = base64.b64encode(video_file.read()).decode('utf-8')

    video_description_prompt = f"""视频描述：\"{caption_summary_text}\"

请执行以下任务：

1. 判断视频中是否包含人物信息。

2. 比较视频描述与视频内容，判断视频描述是否准确反映了视频的内容。

如果视频描述符合上述任务要求，请输出：\"符合\"。

如果视频描述不符合上述任务要求，请输出：\"不符合\"。

请直接输出每个任务结果，不要在回复中提及变量名或额外的信息。
Step by Step Decomposition"""
    print(f"caption_chunk_gpt_compare:{video_description_prompt}")
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
        model="glm-4v-plus",  # 填写需要调用的模型名称
        tools=tools,
        messages=[
          {
            "role": "user",
            "content": [
              {
                "type": "video_url",
                "video_url": {
                    "url" : video_base
                }
              },
              {
                "type": "text",
                "text": video_description_prompt
              }
            ]
          }
        ]
    )

    compare_text = response.choices[0].message.content
    print(f"{video_path}\r\ncaption_chunk_gpt_compare#compare_text:{compare_text}")
    return compare_text


def check_gpt_compare(video_path,compare_text):
    

    video_description_prompt = f"""任务描述：

给定以下文本内容：


{compare_text}


请执行以下操作：

- 判断文本中是否包含关键词 "符合"。
- 如果包含，请输出："是"。
- 如果不包含，请输出："否"。

请直接输出结果，不要包含额外的解释或信息。"""
    print(f"check_gpt_compare:{video_description_prompt}")
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
        model="glm-4-plus",  # 填写需要调用的模型名称
        tools=tools,
        messages=[ 
            {"role": "user", "content": video_description_prompt}
        ]
    )

    check_text = response.choices[0].message.content
    print(f"{video_path}\r\ncheck_gpt_compare#check_text:{check_text}")
    return check_text




def caption_chunk_gpt_role_insert(video_path, features_caption, caption_summary_text):
   
    with open(video_path, 'rb') as video_file:
        video_base = base64.b64encode(video_file.read()).decode('utf-8')

    video_description_prompt = f"""Script Information: {features_caption}

Video Description: "{caption_summary_text}"

Please perform the following tasks:

1. Identify the characters present in both the video and the video description.

2. Compare the video description with the video content and extract the characters from the script to compile a list of roles.

If the video description meets the above requirements, please add the character names to the video description.

If the video description does not meet the above requirements, please output: "Does not meet".

Please output the result directly and do not mention variable names or additional information.

Do not include '\\n' and the word 'video' in your response.

Do not use introductory phrases such as: "The video presents", "The video depicts", "This video showcases", "The video captures", and so on.

Please start the description with the video content directly, such as "A man first sits in a chair, then stands up and walks to the kitchen...."

Do not use phrases like: "as the video progressed" and "Throughout the video".

Please describe the content of the video and the changes that occur, in chronological order.

Please keep the description within 100 English words."""
    print(f"caption_chunk_gpt_role_insert:{video_description_prompt}")
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
        model="glm-4v-plus",  # 填写需要调用的模型名称
        tools=tools,
        messages=[
          {
            "role": "user",
            "content": [
              {
                "type": "video_url",
                "video_url": {
                    "url" : video_base
                }
              },
              {
                "type": "text",
                "text": video_description_prompt
              }
            ]
          }
        ]
    )

    role_caption_text = response.choices[0].message.content
    print(f"{video_path}\r\ncaption_chunk_gpt_role_insert#role_caption_text:{role_caption_text}")
    return role_caption_text


def check_output_folder(output_folder):
    if not os.path.isdir(output_folder):
        print('warning: the folder{} is not exist'.format(output_folder))
        # create srt_folder
        os.makedirs(output_folder)
        print('create folder', output_folder)



 
def check_video_caption_exists_role(root_path, video_source):
    save_path = f'{root_path}/{video_source}/' 
    scene_chunks_csv_path = f'{save_path}/scene_chunks/{video_source}_scene_chunk_frame.csv' 
  
    scene_chunks_role_csv_path = f'{save_path}/scene_chunks/{video_source}_scene_chunk_role_frame.csv' 
    
    df = pd.read_csv(scene_chunks_csv_path)
    # Iterate over each row in the table
    for index, row in df.iterrows():
    
        chunk_path = row['chunk_path']
    
        caption_summary_text = row['caption_summary_text']

        compare_text = caption_chunk_gpt_compare(chunk_path,caption_summary_text)
        
        # Assign the compare_text directly to a new column in the DataFrame
        df.at[index, 'role_compare_text'] = compare_text

        check_text = check_gpt_compare(chunk_path, compare_text)
        # Assign the compare_text directly to a new column in the DataFrame
        df.at[index, 'check_text'] = check_text

        if check_text == '是':
            
            features_caption = row['特征描述']
            role_caption_text = caption_chunk_gpt_role_insert(chunk_path, features_caption, caption_summary_text)
            
            df.at[index, 'role_caption_text'] = role_caption_text


    
    df = df.sort_values(by='特征帧', ascending=True)
    df.to_csv(scene_chunks_role_csv_path, index=False)



import os

root_path = '/mnt/ceph/develop/jiawei/lora_dataset/speech_data/猫和老鼠150/'
for root, dirs, files in os.walk(root_path):
    # 如果你只想获取下一层的子目录，可以在这里筛选
    if root == root_path:
        # root_dir 下的直接子目录就是 dirs 中的项
        for dir in dirs:
            check_video_caption_exists_role(root_path,dir)
    