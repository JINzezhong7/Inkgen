import PyInk
import Texts
import sys
import os
import numpy as np
import torch
import random
sys.path.append(r"../src")
from generate import generate_conditionally
from utilz import offset_points2strokes
import matplotlib.pyplot as plt
from tqdm import tqdm


def Txt2Ink(texts, save_path, prefix, char_to_code_file, model_path, priming_x=None, priming_text="", x_r=None):
    os.makedirs(os.path.join(save_path, 'pngs'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'isfs'), exist_ok=True)
    for i, text in enumerate(tqdm(texts)):
        points = generate_conditionally(text,
                            save_name=os.path.join(save_path,"pngs",f"{prefix}_{i}.png"),
                            device=torch.device('cpu'),
                            cell_size=512,
                            num_clusters=20,
                            K=10,
                            z_size=0,
                            char_to_code_file=char_to_code_file,
                            state_dict_file=model_path,
                            bias=1.0, bias2=1.0,
                            priming_x=priming_x,
                            priming_text=priming_text,
                            x_r=x_r,
                            model_type = 'alexrnn'
        )
        strokes = offset_points2strokes(points)
        inkStrokes = PyInk.InkStrokeList()
        for stroke in strokes:
            inkStrokes.append(np.array(stroke).T)
            # inkStrokes.append(stroke)
        PyInk.SaveISF(inkStrokes, os.path.join(save_path,"isfs",f"{prefix}_{i}.isf"))
        
number_texts = []# [str(i) + " " for i in range(1000)]

# number_texts.append('3.1415926 ') #pi
number_texts.append('0.5 ') #pi
# number_texts.append('2.71828 ') #e
for i in range(100):
    number_texts.append(str(random.random()) + " ")
for i in range(100):
    number_texts.append(str(random.randint(1, 100) + random.random()) + " ")

for i in range(100):
    number_texts.append(str(-random.randint(1, 100) + random.random()) + " ")




Txt2Ink(number_texts, "numbers", "no_priming", '../models/char_to_code.pt', '../models/epoch_119.pt')






    
