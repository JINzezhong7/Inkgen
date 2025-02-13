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
import h5py
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import jiwer


def load_h5_content(h5_file_path, num_samples=100):
    with h5py.File(h5_file_path, 'r') as h5_file:
        contents = h5_file['texts']  
        contents = [c.decode('utf-8').strip()+" " for c in contents]  
    return contents


def Txt2Ink(contents, save_path, prefix, char_to_code_file, model_path, priming_x=None, priming_text="", x_r=None):
    os.makedirs(os.path.join(save_path, 'pngs'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'isfs'), exist_ok=True)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    WERs = []
    with open(os.path.join(save_path, "predictions.txt"), "w") as f:
        for i, content in enumerate(tqdm(contents)):
            # 模型生成笔迹点
            points = generate_conditionally(
                content,
                save_name=os.path.join(save_path, "pngs", f"{prefix}_{i}.png"),
                device=torch.device('cpu'),
                cell_size=512,
                num_clusters=20,
                K=10,
                z_size=0,
                char_to_code_file=char_to_code_file,
                state_dict_file=model_path,
                bias=1.0,
                bias2=1.0,
                priming_x=priming_x,
                priming_text=priming_text,
                x_r=x_r,
                model_type = 'transformer'
            )
            # use ocr to predict
            im = Image.open(os.path.join(save_path, "pngs", f"{prefix}_{i}.png")).convert("RGB")
            pixel_values = processor(im, return_tensors="pt").pixel_values
            generate_ids = model.generate(pixel_values)
            generate_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

            wer = jiwer.wer(content.rstrip(),generate_text)
            WERs.append(wer)

            f.write(f"Image {i};")
            f.write(f"Ground_truth Text: {content.rstrip()};")
            f.write(f"Predicted Text: {generate_text};")
            f.write(f"WER: {wer}\n")

    print(f"Overall WER: {sum(WERs)/len(WERs)}")
        # pred_content = recognize_single_image(im)




if __name__ == "__main__":
   
    h5_file_path = "../datasets/val.h5"  
    char_to_code_file = '../models/char_to_code.pt'
    model_path = '../models/epoch_2.pt'


    contents = load_h5_content(h5_file_path, num_samples=100)
    Txt2Ink(
        contents=contents,
        save_path="../transformer/Iam_first100_full",
        prefix="no_priming",
        char_to_code_file=char_to_code_file,
        model_path=model_path
    )
