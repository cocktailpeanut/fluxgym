"""
UI-free core helpers shared between the Gradio app (app.py) and the RunPod
serverless worker (runpod/handler.py).

app.py builds the entire Gradio Blocks at import time, so the serverless handler
cannot import it. Everything here is pure logic (model registry, path helpers,
model download, dataset creation, README generation) with no Gradio dependency,
so both entrypoints can import it safely.
"""
import os
import re
import shutil
import yaml
from PIL import Image
from slugify import slugify
from huggingface_hub import hf_hub_download

# Load the model registry relative to this file (robust regardless of cwd).
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_CORE_DIR, "models.yaml"), "r") as _f:
    models = yaml.safe_load(_f)


def resolve_path(p):
    """Absolute, shell-quoted path under the repo root (for train.sh)."""
    norm_path = os.path.normpath(os.path.join(_CORE_DIR, p))
    return f"\"{norm_path}\""


def resolve_path_without_quotes(p):
    """Absolute path under the repo root, no quoting (for filesystem ops)."""
    return os.path.normpath(os.path.join(_CORE_DIR, p))


def readme(base_model, lora_name, instance_prompt, sample_prompts):
    # model license
    model_config = models[base_model]
    model_file = model_config["file"]
    base_model_name = model_config["base"]
    license = None
    license_name = None
    license_link = None
    license_items = []
    if "license" in model_config:
        license = model_config["license"]
        license_items.append(f"license: {license}")
    if "license_name" in model_config:
        license_name = model_config["license_name"]
        license_items.append(f"license_name: {license_name}")
    if "license_link" in model_config:
        license_link = model_config["license_link"]
        license_items.append(f"license_link: {license_link}")
    license_str = "\n".join(license_items)
    print(f"license_items={license_items}")
    print(f"license_str = {license_str}")

    # tags
    tags = ["text-to-image", "flux", "lora", "diffusers", "template:sd-lora", "fluxgym"]

    # widgets
    widgets = []
    sample_image_paths = []
    output_name = slugify(lora_name)
    samples_dir = resolve_path_without_quotes(f"outputs/{output_name}/sample")
    try:
        for filename in os.listdir(samples_dir):
            # Filename Schema: [name]_[steps]_[index]_[timestamp].png
            match = re.search(r"_(\d+)_(\d+)_(\d+)\.png$", filename)
            if match:
                steps, index, timestamp = int(match.group(1)), int(match.group(2)), int(match.group(3))
                sample_image_paths.append((steps, index, f"sample/{filename}"))

        # Sort by numeric index
        sample_image_paths.sort(key=lambda x: x[0], reverse=True)

        final_sample_image_paths = sample_image_paths[:len(sample_prompts)]
        final_sample_image_paths.sort(key=lambda x: x[1])
        for i, prompt in enumerate(sample_prompts):
            _, _, image_path = final_sample_image_paths[i]
            widgets.append(
                {
                    "text": prompt,
                    "output": {
                        "url": image_path
                    },
                }
            )
    except:
        print(f"no samples")
    dtype = "torch.bfloat16"
    # Construct the README content
    readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model_name}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
{license_str}
---

# {lora_name}

A Flux LoRA trained on a local computer with [Fluxgym](https://github.com/cocktailpeanut/fluxgym)

<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, Forge, etc.

Weights for this model are available in Safetensors format.

"""
    return readme_content


def download(base_model, info=print):
    """Download the base model UNet + shared VAE/CLIP/T5 if not already present.

    `info` is a single-argument logging callback (defaults to print). app.py
    passes gr.Info so downloads surface as UI toasts; the worker passes print.
    """
    model = models[base_model]
    model_file = model["file"]
    repo = model["repo"]

    # download unet
    if base_model == "flux-dev" or base_model == "flux-schnell":
        unet_folder = "models/unet"
    else:
        unet_folder = f"models/unet/{repo}"
    unet_path = os.path.join(unet_folder, model_file)
    if not os.path.exists(unet_path):
        os.makedirs(unet_folder, exist_ok=True)
        info(f"Downloading base model: {base_model}. Please wait. (You can check the terminal for the download progress)")
        print(f"download {base_model}")
        hf_hub_download(repo_id=repo, local_dir=unet_folder, filename=model_file)

    # download vae
    vae_folder = "models/vae"
    vae_path = os.path.join(vae_folder, "ae.sft")
    if not os.path.exists(vae_path):
        os.makedirs(vae_folder, exist_ok=True)
        info(f"Downloading vae")
        print(f"downloading ae.sft...")
        hf_hub_download(repo_id="cocktailpeanut/xulf-dev", local_dir=vae_folder, filename="ae.sft")

    # download clip
    clip_folder = "models/clip"
    clip_l_path = os.path.join(clip_folder, "clip_l.safetensors")
    if not os.path.exists(clip_l_path):
        os.makedirs(clip_folder, exist_ok=True)
        info(f"Downloading clip...")
        print(f"download clip_l.safetensors")
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="clip_l.safetensors")

    # download t5xxl
    t5xxl_path = os.path.join(clip_folder, "t5xxl_fp16.safetensors")
    if not os.path.exists(t5xxl_path):
        print(f"download t5xxl_fp16.safetensors")
        info(f"Downloading t5xxl...")
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="t5xxl_fp16.safetensors")


def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size / width) * height)
        else:
            new_height = size
            new_width = int((size / height) * width)
        print(f"resize {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)


def create_dataset(destination_folder, size, *inputs):
    print("Creating dataset")
    images = inputs[0]
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, image in enumerate(images):
        # copy the images to the datasets folder
        new_image_path = shutil.copy(image, destination_folder)

        # if it's a caption text file skip the next bit
        ext = os.path.splitext(new_image_path)[-1].lower()
        if ext == '.txt':
            continue

        # resize the images
        resize_image(new_image_path, new_image_path, size)

        # copy the captions

        original_caption = inputs[index + 1]

        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = resolve_path_without_quotes(os.path.join(destination_folder, caption_file_name))
        print(f"image_path={new_image_path}, caption_path = {caption_path}, original_caption={original_caption}")
        # if caption_path exists, do not write
        if os.path.exists(caption_path):
            print(f"{caption_path} already exists. use the existing .txt file")
        else:
            print(f"{caption_path} create a .txt caption file")
            with open(caption_path, 'w') as file:
                file.write(original_caption)

    print(f"destination_folder {destination_folder}")
    return destination_folder
