import gradio as gr
from PIL import Image
import torch
from diffusers import DiffusionPipeline, AutoencoderTiny
import os

SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", None)
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", None)

if SAFETY_CHECKER:
    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        custom_pipeline="lcm_txt2img",
        scheduler=None,
    )
else:
    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        custom_pipeline="lcm_txt2img",
        scheduler=None,
        safety_checker=None,
    )
pipe.to(device="cuda", dtype=torch.float16)
pipe.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", device="cuda", torch_dtype=torch.float16
)
pipe.vae = pipe.vae.cuda()
pipe.unet.to(memory_format=torch.channels_last)
pipe.set_progress_bar_config(disable=True)

if TORCH_COMPILE:
    pipe.text_encoder = torch.compile(pipe.text_encoder, mode="max-autotune")
    pipe.tokenizer = torch.compile(pipe.tokenizer, mode="max-autotune")
    pipe.unet = torch.compile(pipe.unet, mode="max-autotune")
    pipe.vae = torch.compile(pipe.vae, mode="max-autotune")


def predict(prompt1, prompt2, merge_ratio, guidance, steps, sharpness, seed=1231231):
    torch.manual_seed(seed)
    img = pipe(
        prompt1=prompt1,
        prompt2=prompt2,
        sv=merge_ratio,
        sharpness=sharpness,
        width=512,
        height=512,
        num_inference_steps=steps,
        guidance_scale=guidance,
        lcm_origin_steps=50,
        output_type="pil",
        return_dict=False,
    )
    return img


css="""
#container{
    margin: 0 auto;
    max-width: 80rem;
}
#intro{
    max-width: 32rem;
    text-align: center;
    margin: 0 auto;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown(
            """# SDZoom

    Welcome to sdzoom, a testbed application designed for optimizing and experimenting with various 
    configurations to achieve the fastest Stable Diffusion (SD) pipelines.
    RTSD leverages the expertise provided by Latent Consistency Models (LCM). For more information about LCM,
    visit their website at [Latent Consistency Models](https://latent-consistency-models.github.io/).

    """, elem_id="intro"
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil")
            with gr.Column():
                merge_ratio = gr.Slider(
                    value=50, minimum=1, maximum=100, step=1, label="Merge Ratio"
                )
                guidance = gr.Slider(
                    label="Guidance", minimum=1, maximum=50, value=10.0, step=0.01
                )
                steps = gr.Slider(label="Steps", value=4, minimum=2, maximum=20, step=1)
                sharpness = gr.Slider(
                    value=1.0, minimum=0, maximum=1, step=0.001, label="Sharpness"
                )
                seed = gr.Slider(randomize=True, minimum=0, maximum=12013012031030, label="Seed")
                prompt1 = gr.Textbox(label="Prompt 1")
                prompt2 = gr.Textbox(label="Prompt 2")
                generate_bt = gr.Button("Generate")

        inputs = [prompt1, prompt2, merge_ratio, guidance, steps, sharpness, seed]
        gr.Examples(
            examples=[
                ["Elon Musk", "Mark Zuckerberg", 50, 10.0, 4, 1.0, 1231231],
                ["Elon Musk", "Bill Gates", 50, 10.0, 4, 1.0, 53453],
                [
                    "Asian women, intricate jewlery in her hair, 8k",
                    "Tom Cruise, intricate jewlery in her hair, 8k",
                    50,
                    10.0,
                    4,
                    1.0,
                    542343,
                ],
            ],
            fn=predict,
            inputs=inputs,
            outputs=image,
        )
        generate_bt.click(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        seed.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        merge_ratio.change(
            fn=predict, inputs=inputs, outputs=image, show_progress=False
        )
        guidance.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        steps.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        sharpness.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        prompt1.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        prompt2.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)

demo.queue()
if __name__ == "__main__":
    demo.launch()
