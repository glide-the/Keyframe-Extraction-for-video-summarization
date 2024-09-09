import math
from typing import Union, List
 
import os
from datetime import datetime
import numpy as np
import itertools
import PIL.Image 
import tqdm
import logging  

logger = logging.getLogger(__file__)
   
 
 
def save_video(tensor: Union[List[np.ndarray], List[PIL.Image.Image]], fps: int = 8):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path, fps=fps)
    return video_path


class ProgressBar:
    def __init__(self, total, desc=None):
        self.total = total
        self.current = 0
        self.b_unit = tqdm.tqdm(total=total, desc="ProgressBar context index: 0" if desc is None else desc)

    def update(self, value):
        if value > self.total:
            value = self.total
        self.current = value
        if self.b_unit is not None:
            self.b_unit.set_description("ProgressBar context index: {}".format(self.current))
            self.b_unit.refresh()

            # 更新进度
            self.b_unit.update(self.current)
