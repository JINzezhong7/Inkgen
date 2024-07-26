import argparse
from diffusers import DiffusionPipeline
import os
import torch

def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_dir")
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype = torch.float16, use_safetensors = True, variant = "fp16")
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)
    pipe.save_pretrained(args.model_output_dir)

if __name__ == "__main__":
    main()