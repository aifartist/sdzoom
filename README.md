# sdzoom
# Testbed for optimizing the fastest SD pipelines.

# Application #1
#
# RTSD - Real Time Stable Diffusion
# verson 0.0.0.alpha
#
# Powered by LCM - https://github.com/0xbitches/sd-webui-lcm
#
# It is the seed of an idea and this is two days of coding
# trying to learn html/javascript.  I have many ideas for
# adding far more knobs and getting real time feedback.

# Some of the following are Linux centric.  For Windows,
# you need to know how to setup a Windows venv.

# Setup
python3 -m venv `pwd`/venv
source venv/bin/activate
pip3 install -r requirements.txt

# Running it
python3 rtsd.py

# Connect your browser to 127.0.0.1:5017

# Usage
As you move the sliders the image will update immediately.
Put in two prompts like "cat" and "dog", or "Emma Watson"
and "Tom Cruise".  As you slide Merge Ratio you'll blend
these prompts.  Guidance and steps are typical to SD. More
steps improve the image.

The seed if fixed to a initial value when you start the app.
This produces better consistancy to compare images as you
adjust the sliders.  However, you can click "New seed" to
change the seed.
