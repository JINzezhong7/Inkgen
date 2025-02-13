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
import string
def Txt2Ink(texts, save_path, prefix, char_to_code_file, model_path, priming_x=None, priming_text="", x_r=None):
    os.makedirs(os.path.join(save_path, 'pngs'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'isfs'), exist_ok=True)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    CERs = []
    with open(os.path.join(save_path, "predictions.txt"), "w") as f:
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
            # use ocr to predict
            im = Image.open(os.path.join(save_path, "pngs", f"{prefix}_{i}.png")).convert("RGB")
            pixel_values = processor(im, return_tensors="pt").pixel_values
            generate_ids = model.generate(pixel_values)
            generate_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

            cleaned_text = generate_text.replace(" ","")
            cer = jiwer.cer(text.rstrip(),cleaned_text)
            CERs.append(cer)

            f.write(f"Image {i};")
            f.write(f"Ground_truth Text: {text.rstrip()};")
            f.write(f"Predicted Text: {cleaned_text};")
            f.write(f"CER: {cer}\n")

    print(f"Overall CER: {sum(CERs)/len(CERs)}")


def generate_random_texts(num_samples=100):
    texts = []
    
    # 生成地址
    for _ in range(num_samples // 2):
        street_number = random.randint(1, 9999)
        street_name = ''.join(random.choices(string.ascii_letters, k=random.randint(5, 10)))
        street_suffix = random.choice(["St", "Ave", "Blvd", "Rd", "Ln"])
        city = ''.join(random.choices(string.ascii_letters, k=random.randint(4, 8)))
        state = ''.join(random.choices(string.ascii_uppercase, k=2))
        zip_code = random.randint(10000, 99999)
        address = f"{street_number} {street_name} {street_suffix}, {city}, {state} {zip_code}"
        texts.append(address + " ")
    
    # 生成人名地名
    for _ in range(num_samples // 2):
        first_name = ''.join(random.choices(string.ascii_letters, k=random.randint(3, 8)))
        last_name = ''.join(random.choices(string.ascii_letters, k=random.randint(5, 10)))
        place_name = ''.join(random.choices(string.ascii_letters, k=random.randint(4, 8)))
        full_name_or_place = random.choice([
            f"{first_name} {last_name}", 
            f"{place_name} City"
        ])
        texts.append(full_name_or_place + " ")
    
    return texts

def generate_random_decimals(num_samples, min_digits=10, max_digits=16):
    """
    生成一组 10 到 16 位的随机小数，范围在 0 到 1 之间。

    Args:
        num_samples (int): 生成的小数数量。
        min_digits (int): 最小的有效位数。
        max_digits (int): 最大的有效位数。

    Returns:
        list: 包含随机小数的字符串列表。
    """
    random_decimals = []
    for _ in range(num_samples):
        num_digits = random.randint(min_digits, max_digits)  # 确定位数
        decimal_part = random.randint(10**(num_digits-1), 10**num_digits - 1)  # 小数部分的有效数字
        random_decimal = f"0.{decimal_part}"
        random_decimals.append(random_decimal + " ")
    return random_decimals

def generate_random_integers(num_samples, min_digits=1, max_digits=3):
    """
    生成一组 5 到 9 位的随机整数。

    Args:
        num_samples (int): 生成的整数数量。
        min_digits (int): 最小的位数。
        max_digits (int): 最大的位数。

    Returns:
        list: 包含随机整数的字符串列表。
    """
    random_integers = []
    for _ in range(num_samples):
        num_digits = random.randint(min_digits, max_digits)  # 确定位数
        random_integer = random.randint(10**(num_digits-1), 10**num_digits - 1)  # 生成对应位数的整数
        random_integers.append(str(random_integer)+ " ")
    return random_integers

if __name__ == "__main__":
    number_texts = generate_random_decimals(num_samples=100)

    Txt2Ink(number_texts, "../repo_results/numbers", "no_priming", '../models/char_to_code.pt', '../models/epoch_119.pt')






    
