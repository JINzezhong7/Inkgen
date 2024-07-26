import argparse
import copy
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, AutoencoderKL
import os
import shutil
import torch


T2IAdapterName = "TencentARC/t2i-adapter-sketch-sdxl-1.0"
VAEName = "madebyollin/sdxl-vae-fp16-fix"
PREPROCESSORSName = "preprocessors"
SDXLName = "stabilityai/stable-diffusion-xl-base-1.0"
LORAName = "lora"

def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hednet_model_dir")
    parser.add_argument("--lora_model_dir")
    parser.add_argument("--model_output_dir")
    return parser

def AddPrefix(state_dict, prefix):
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
        state_dict_v2[prefix +'.'+key] = state_dict_v2.pop(key)
    return state_dict_v2

def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    adapter = T2IAdapter.from_pretrained(T2IAdapterName, torch_dtype = torch.float16,local_files_only = False)
    vae = AutoencoderKL.from_pretrained(VAEName, local_files_only = False, torch_dtype = torch.float16)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        SDXLName, vae=vae, adapter=adapter, torch_dtype = torch.float16, local_files_only = False
    )

    model_output_dir = os.path.join(args.model_output_dir, "models")

    if not os.path.exists(os.path.join(model_output_dir, T2IAdapterName )):
        os.makedirs(os.path.join(model_output_dir, T2IAdapterName ))

    if not os.path.exists(os.path.join(model_output_dir, VAEName )):
        os.makedirs(os.path.join(model_output_dir, VAEName ))

    if not os.path.exists(os.path.join(model_output_dir, SDXLName )):
        os.makedirs(os.path.join(model_output_dir, SDXLName ))

    if not os.path.exists(os.path.join(model_output_dir, PREPROCESSORSName )):
        os.makedirs(os.path.join(model_output_dir, PREPROCESSORSName ))

    if not os.path.exists(os.path.join(model_output_dir, LORAName )):
        os.makedirs(os.path.join(model_output_dir, LORAName ))

    # only copy .safetensors files
    for (_, _, files) in os.walk(args.lora_model_dir,topdown = True):
        for name in files:
            if ".safetensors" in name:
                shutil.copy(os.path.join(args.lora_model_dir, name), os.path.join(model_output_dir, LORAName + "/" + name))

    pipe.save_pretrained(os.path.join(model_output_dir, SDXLName))
    vae.save_pretrained(os.path.join(model_output_dir, VAEName))
    adapter.save_pretrained(os.path.join(model_output_dir, T2IAdapterName))
    # only copy .pth files
    for (_, _, files) in os.walk(args.hednet_model_dir,topdown = True):
        for name in files:
            if ".pth" in name:
                shutil.copy(os.path.join(args.hednet_model_dir, name), os.path.join(model_output_dir, PREPROCESSORSName + "/" + name))


if __name__ == "__main__":
    main()