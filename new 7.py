from diffusers import StableVideoDiffusionPipeline

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Generate video from an initial image
image = Image.open("initial_frame.png")
frames = pipe(
    image,
    num_frames=25,
    decode_chunk_size=5  # Reduce memory usage
).frames[0]