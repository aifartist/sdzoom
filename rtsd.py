import io, os
import threading
import time
import random
import base64

from flask import Flask, request, jsonify, render_template, session, abort
import logging
import requests
import secrets

import torch
from diffusers import DiffusionPipeline

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="lcm_txt2img", scheduler=None, safety_checker=None)

pipe.to(torch_device="cuda", torch_dtype=torch.float16)

from diffusers import AutoencoderTiny
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesd', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()

pipe.unet.to(memory_format=torch.channels_last) # Helps on 4090 YMMV

if False:
    pipe.text_encoder = torch.compile(pipe.text_encoder, mode='max-autotune')
    pipe.tokenizer = torch.compile(pipe.tokenizer, mode='max-autotune')

    pipe.unet = torch.compile(pipe.unet, mode='max-autotune')

    pipe.vae = torch.compile(pipe.vae, mode='max-autotune')


serverState = 1
p1 = "Asian women, intricate jewlery in her hair, 8k"
p2 = "Tom Cruise, intricate jewlery in her hair, 8k"
seed = 12321
slider_value = 50

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

app = Flask(__name__, template_folder=os.path.abspath('.'))
app.secret_key = secrets.token_hex(16)

def logit(msg):
    print(f"{time.time()}: {request.remote_addr} {msg}")

def send_post_request(url, payload):
    # Send an HTTP POST request with the payload
    response = requests.post(url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        try:
            json_response = response.json()
            return json_response
        except json.JSONDecodeError:
            logit("Invalid JSON response")
    else:
        logit(f"Request failed with status code {response.status_code}")
    
    return None

@app.route('/notification', methods=['GET'])
def handle_get_request():
    return notification

@app.route('/')
def index():
    session['user'] = None
    logit(f"New connection from {request.remote_addr}")
    return render_template('index.html')

sessions = {}

@app.before_request
def before_post_request():
    if request.method == 'POST':
        if serverState == 0:
            abort(jsonify({'success': False, 'message': 'Server is offline'}))


@app.route('/newseed', methods=['POST'])
def register():
    global seed

    seed = random.randint(0, 2100000000)
    torch.manual_seed(seed)

    response = {'success': True}

    return jsonify(response)

@app.route('/update_slider', methods=['POST'])
def update_slider():
    xxx = 1/0
    global slider_value, p1, p2
    data = request.get_json()
    slider_name = data['name']
    slider_value = int(data['value'])
    # Process the slider data as needed
    print(f"\nReceived update for {slider_name}: {slider_value}\n")
    torch.manual_seed(12321)
    img = pipe(prompt1=p1, prompt2=p2, sv=slider_value,
        width=512, height=512,
        num_inference_steps=4,
        guidance_scale=10., lcm_origin_steps=50,
        output_type="pil", return_dict=False)

    imgBytes = io.BytesIO()
    img.save(imgBytes, format="JPEG")
    #img.save("dwgold.jpg")
    b64img = base64.b64encode(imgBytes.getvalue()).decode('utf-8')
    response = {'image': b64img}
    return jsonify(response)
    #return 'Slider updated'

@app.route('/notify4090', methods=['POST'])
def notify():
    global notification
    data = request.get_json()
    notification = data['notification']
    print(f"{time.time()}: Notification set to {notification}")
    return f"Notification set to {notification}"

@app.route('/state4090', methods=['POST'])
def state():
    global serverState
    data = request.get_json()
    state = data['state']
    print(f"{time.time()}: State set to {state}")
    return f"state set to {state}"

@app.route('/submit', methods=['POST'])
def submit():
    global seed, mergeRatio
    print('IN: SUBMIT')
    data = request.get_json()
    print(f"data = {data}")

    newSeed = data['newSeed']
    p1 = data['prompt']
    p2 = data['negative']
    mergeRatio = int(data['mergeRatio'])
    width = data['width']
    height = data['height']
    nSteps = int(data['nSteps'])
    guidance = float(data['guidance'])
    sharpness = float(data['sharpness'])

    print(f"newSeed = {newSeed}")
    print(f"nSteps = {nSteps}")
    print(f"mergeRation = {mergeRatio}")
    print(f"guidance = {guidance}")
    print(f"sharpness = {sharpness}")

    if int(newSeed) == 1:
        seed = random.randint(0, 2100000000)

    torch.manual_seed(seed)

    #try:
    tm0 = time.time()
    img = pipe(prompt1=p1, prompt2=p2, sv=mergeRatio,
        sharpness=sharpness,
        width=512, height=512,
        num_inference_steps=nSteps,
        guidance_scale=guidance, lcm_origin_steps=50,
        output_type="pil", return_dict=False)
    print(f"time = {time.time() - tm0}")
    #except:
    #    return jsonify({ 'message': 'Generate image failed' })

    imgBytes = io.BytesIO()
    img.save(imgBytes, format="JPEG")
    #img.save("dwgold.jpg")
    b64img = base64.b64encode(imgBytes.getvalue()).decode('utf-8')
    response = {'image': b64img}
    print('OUT: SUBMIT')
    return jsonify(response)

    #if b64Image:
    #    response = {'image': b64img}
    #else:
    #    response = {'message': 'Image generation failed'}

    #return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5017)
