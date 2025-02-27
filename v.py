Python 3.12.0 (tags/v3.12.0:0fb18b0, Oct  2 2023, 13:03:39) [MSC v.1935 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image
import subprocess
import os

# Configuration
VIDEO_LENGTH_SEC = 5  # 5-second video
FPS = 12  # Lower FPS reduces computation
WIDTH, HEIGHT = 512, 512  # Resolution
PROMPT = "haunted mansion at midnight, horror, eerie atmosphere, 4k, realistic"
OUTPUT_VIDEO = "horror_video.mp4"
OUTPUT_AUDIO = "horror_audio.wav"
FINAL_OUTPUT = "final_horror_video.mp4"

# Step 1: Generate Frames with Stable Diffusion
def generate_frames():
    # Load model (requires GPU and ~10GB VRAM)
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        safety_checker=None  # Disable if NSFW filter blocks horror content
    ).to("cuda")

    # Generate frames with slight prompt variations
    frames = []
    for i in range(VIDEO_LENGTH_SEC * FPS):
        # Adjust prompt for variation (e.g., "full moon appears")
        frame_prompt = PROMPT + f", scene {i//6}"  # Change scene every 0.5 seconds
        
        # Generate image
        image = pipe(
            prompt=frame_prompt,
            width=WIDTH,
...             height=HEIGHT
...         ).images[0]
...         
...         # Convert to OpenCV format
...         frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
...         frames.append(frame)
...     
...     return frames
... 
... # Step 2: Assemble Frames into Video
... def create_video(frames):
...     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
...     video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))
...     
...     for frame in frames:
...         video.write(frame)
...     video.release()
... 
... # Step 3: Add Audio (FFmpeg required)
... def add_audio():
...     # Generate or download a horror audio track (example uses FFmpeg to add silence)
...     os.system(f"ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -t {VIDEO_LENGTH_SEC} {OUTPUT_AUDIO}")
...     
...     # Merge audio and video
...     os.system(
...         f"ffmpeg -i {OUTPUT_VIDEO} -i {OUTPUT_AUDIO} "
...         f"-c:v copy -c:a aac -strict experimental {FINAL_OUTPUT}"
...     )
... 
... # Run Pipeline
... if __name__ == "__main__":
...     print("Generating frames... (This may take 30+ minutes)")
...     frames = generate_frames()
...     print("Creating video...")
...     create_video(frames)
...     print("Adding audio...")
...     add_audio()
