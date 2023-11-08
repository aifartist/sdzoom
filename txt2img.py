import time
import random
import os, signal
import subprocess
from subprocess import check_output

import torch
from diffusers import DiffusionPipeline

doCompile = False
# For Linux: Use at your own risk.
goCrazy = False

torch.set_default_device('cuda')
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 0

if not os.path.exists('txt2img'):
    os.makedirs('txt2img')

from PIL import Image
def save_image(steps, seq, images, image_path_dir, image_name_prefix):
    idx = 0
    for img in images:
        img.save(f"txt2img/{image_name_prefix}.jpg")
        idx += 1

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline=f"clean_txt2img", scheduler=None, safety_checker=None)

pipe.to(torch_device="cuda", torch_dtype=torch.float16)

pipe.unet.to(memory_format=torch.channels_last)

from diffusers import AutoencoderTiny
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesd', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()

if doCompile:
    pipe.text_encoder = torch.compile(pipe.text_encoder, mode='max-autotune')
    pipe.tokenizer = torch.compile(pipe.tokenizer, mode='max-autotune')
    pipe.unet = torch.compile(pipe.unet, mode='max-autotune')
    pipe.vae = torch.compile(pipe.vae, mode='max-autotune')

prompt = "Asian women wearing fancy dress, intricate jewlery in her hair, 8k"
nSteps = 4
guidance = 10.0

# Warmup
print('\nStarting warmup of two images.  If using compile()')
print('    this can take an extra 35 seconds each time.')
with torch.inference_mode():
    img = pipe(prompt=prompt, width=512, height=512,
        num_inference_steps=nSteps,
        guidance_scale=guidance, lcm_origin_steps=50,
        output_type="pil", return_dict=False)
    img = pipe(prompt=prompt, width=512, height=512,
        num_inference_steps=nSteps,
        guidance_scale=guidance, lcm_origin_steps=50,
        output_type="pil", return_dict=False)

seed = random.randint(0, 2147483647)
torch.manual_seed(seed)

with torch.inference_mode():
    try:
        # This STOP'ing of processes idles enough cores to allow
        # one core to hit 5.8 GHz and CPU perf does matter if you
        # have a 4090.
        # Warning: Stopping gnome-shell freezes the desktop until the
        # gen is done.  If you control-c you may have to hard restart.
        if goCrazy:
            subprocess.Popen('pkill -STOP chrome', shell=True)
            pid = int(check_output(["pidof", '-s', '/usr/bin/gnome-shell']))
            os.kill(pid, signal.SIGSTOP)

        tm0 = time.time()
        for idx in range(0, 100):
            tm00 = time.time()
            img = pipe(prompt=prompt, width=512, height=512,
                num_inference_steps=nSteps,
                guidance_scale=guidance, lcm_origin_steps=50,
                output_type="pil", return_dict=False)[0]
            print(f"time = {(1000*(time.time() - tm00)):5.2f} milliseconds")
            img[0].save(f"txt2img/{idx:05d}-t2i.jpg")
        print(f"time = {time.time() - tm0}")
    except Exception as e:
        raise e
    finally:
        pass
        if goCrazy:
            os.kill(pid, signal.SIGCONT)
            subprocess.Popen('pkill -CONT chrome', shell=True)
