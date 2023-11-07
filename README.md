# sdzoom: Optimize Your Stable Diffusion Pipelines

Welcome to sdzoom, a testbed application designed for optimizing and experimenting with various configurations to achieve the fastest Stable Diffusion (SD) pipelines.

## RTSD - Real Time Stable Diffusion (v0.0.0.alpha)

RTSD is an application that enables real-time interactions with LCM models. This project is in its early stages and represents a concept brought to life with a couple of days of coding aimed at getting acquainted with HTML and Javascript. The ambition is to introduce more controls and achieve real-time feedback for users.

RTSD leverages the expertise provided by Latent Consistency Models (LCM). For more information about LCM, visit their website at [Latent Consistency Models](https://latent-consistency-models.github.io/).

### Setup Instructions

To get started with RTSD, you will need to set up a Python virtual environment and install the necessary dependencies. Below are step-by-step instructions tailored for Linux environments. If you're on Windows, please ensure you're familiar with setting up a virtual environment (venv) on that platform.

```bash
# Create a virtual environment in the current directory
python3 -m venv ./venv

# Activate the virtual environment
source venv/bin/activate

# Install the required dependencies
pip3 install -r requirements.txt
```

### Running the Application

To run RTSD, execute the following command, and ensure you have activated the virtual environment as described in the setup instructions.

```bash
python3 rtsd.py
```

After starting the application, connect to it using your web browser by visiting `http://127.0.0.1:5017`.

### Usage Guide

RTSD offers an interactive experience where image adjustments are reflected in real-time. Here’s how to use it:

- Move the sliders, and the image will update instantaneously.
- Provide two prompts, such as "cat" and "dog" or "Emma Watson" and "Tom Cruise".
- Use the "Merge Ratio" slider to blend these prompts.
- Adjust "Guidance" and the number of "Steps" as you would typically do with SD. More steps generally lead to a clearer image.

RTSD initializes with a fixed seed value, providing consistency that allows for better comparison of images as you tweak settings. If you wish to use a different seed, you can click the "New Seed" button to generate a new one.

Enjoy experimenting with RTSD, and we welcome your feedback and ideas to enhance this tool further!
