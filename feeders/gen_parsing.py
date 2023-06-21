import os
import numpy as np
import PIL.Image as Image
import math
import random
from torchvision import transforms

def post_process(pred):
    img = Image.fromarray(pred.squeeze(-1))
    return img

def gen_featuremap(filename, palette, random_interval = False, temporal_rgb_frames = 9, aug=False):
    New_image = Image.new('RGB', size = (480, 480))
    pred_path = '/root/liujinfu/Self-Correction-Human-Parsing/output/' + filename + '.npy'
    # pred_path = 'Parsing/outputs/' + filename + '.npy'
    if not os.path.exists(pred_path):
        raise RuntimeError("Parsing path is not exists!!")
    
    data = np.load(pred_path, allow_pickle = True)
    num_frames = data.shape[0]

    if num_frames < temporal_rgb_frames:
        data = np.pad(data, ((0, temporal_rgb_frames-num_frames),(0,0),(0,0),(0,0)), 'reflect')
        num_frames = data.shape[0]

    start = 0
    sample_interval = num_frames // temporal_rgb_frames
    if random_interval: 
        start = np.random.randint(0, num_frames % temporal_rgb_frames + 1)
        sample_interval = np.random.randint(1, num_frames // temporal_rgb_frames + 1)
    
    frame_range = range(start, num_frames, sample_interval)

    hflip = 0
    r_angle = 0
    if aug:
        hflip = random.random()
        r_angle = random.random() * 30

    for idx, value in enumerate(frame_range[0:temporal_rgb_frames]):
        sqrt_frames = math.sqrt(temporal_rgb_frames)
        img = post_process(data[value]).resize((int(480/sqrt_frames), (int(480/sqrt_frames))))
        img.putpalette(palette)
        if aug:
            if hflip > 0.5:
                img = transforms.functional.hflip(img)
            img = transforms.functional.rotate(img, r_angle)
        New_image.paste(img, (int(480/sqrt_frames)*int(idx%sqrt_frames), int(480/sqrt_frames)*int(idx//sqrt_frames)))

    return New_image