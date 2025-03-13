# Flux Gym

Dead simple web UI for training FLUX LoRA **with LOW VRAM (12GB/16GB/20GB) support.**

- **Frontend:** The WebUI forked from [AI-Toolkit](https://github.com/ostris/ai-toolkit) (Gradio UI created by https://x.com/multimodalart)
- **Backend:** The Training script powered by [Kohya Scripts](https://github.com/kohya-ss/sd-scripts)

FluxGym supports 100% of Kohya sd-scripts features through an [Advanced](#advanced) tab, which is hidden by default.

![screenshot.png](screenshot.png)

---


# What is this?

1. I wanted a super simple UI for training Flux LoRAs
2. The [AI-Toolkit](https://github.com/ostris/ai-toolkit) project is great, and the gradio UI contribution by [@multimodalart](https://x.com/multimodalart) is perfect, but the project only works for 24GB VRAM.
3. [Kohya Scripts](https://github.com/kohya-ss/sd-scripts) are very flexible and powerful for training FLUX, but you need to run in terminal.
4. What if you could have the simplicity of AI-Toolkit WebUI and the flexibility of Kohya Scripts?
5. Flux Gym was born. Supports 12GB, 16GB, 20GB VRAMs, and extensible since it uses Kohya Scripts underneath.

---

# News

- September 25: Docker support + Autodownload Models (No need to manually download models when setting up) + Support custom base models (not just flux-dev but anything, just need to include in the [models.yaml](models.yaml) file.
- September 16: Added "Publish to Huggingface" + 100% Kohya sd-scripts feature support: https://x.com/cocktailpeanut/status/1835719701172756592
- September 11: Automatic Sample Image Generation + Custom Resolution: https://x.com/cocktailpeanut/status/1833881392482066638

---

# Supported Models

1. Flux1-dev
2. Flux1-dev2pro (as explained here: https://medium.com/@zhiwangshi28/why-flux-lora-so-hard-to-train-and-how-to-overcome-it-a0c70bc59eaf)
3. Flux1-schnell (Couldn't get high quality results, so not really recommended, but feel free to experiment with it)
4. More?

The models are automatically downloaded when you start training with the model selected.

You can easily add more to the supported models list by editing the [models.yaml](models.yaml) file. If you want to share some interesting base models, please send a PR.

---

# How people are using Fluxgym

Here are people using Fluxgym to locally train Lora sharing their experience:

https://pinokio.computer/item?uri=https://github.com/cocktailpeanut/fluxgym


# More Info

To learn more, check out this X thread: https://x.com/cocktailpeanut/status/1832084951115972653

# Install

## 1. One-Click Install

You can automatically install and launch everything locally with Pinokio 1-click launcher: https://pinokio.computer/item?uri=https://github.com/cocktailpeanut/fluxgym


## 2. Install Manually

First clone Fluxgym and kohya-ss/sd-scripts:

```
git clone https://github.com/cocktailpeanut/fluxgym
cd fluxgym
git clone -b sd3 https://github.com/kohya-ss/sd-scripts
```

Your folder structure will look like this:

```
/fluxgym
  app.py
  requirements.txt
  /sd-scripts
```

Now activate a venv from the root `fluxgym` folder:

If you're on Windows:

```
python -m venv env
env\Scripts\activate
```

If your're on Linux:

```
python -m venv env
source env/bin/activate
```

This will create an `env` folder right below the `fluxgym` folder:

```
/fluxgym
  app.py
  requirements.txt
  /sd-scripts
  /env
```

Now go to the `sd-scripts` folder and install dependencies to the activated environment:

```
cd sd-scripts
pip install -r requirements.txt
```

Now come back to the root folder and install the app dependencies:

```
cd ..
pip install -r requirements.txt
```

Finally, install pytorch Nightly:

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


# Start

## 1. Start via `app-launch.sh`

For Linux users, you can use the `app-launch.sh` script to start the app.

```
sh app-launch.sh
# or give execute permission and run
# chmod +x app-launch.sh
./app-launch.sh
```
Also, you can modify the `app-launch.sh` script to set the environment variables as you want.

```
# multi-thread download
export HF_HUB_ENABLE_HF_TRANSFER="1"

# disable analytics
export GRADIO_ANALYTICS_ENABLED="0"

# HuggingFace cache directory
#export HF_HOME="/path/to/your/custom/cache"

# For HuggingFace mirror site to accelerate download for China users
export HF_ENDPOINT="https://hf-mirror.com"

# Set directory path by environment variables
export OUTPUTS_DIR="/root/autodl-tmp/fluxgym/outputs"
export DATASETS_DIR="/root/autodl-tmp/fluxgym/datasets"
export MODELS_DIR="/root/autodl-tmp/fluxgym/models"

# Set file path by environment variables
export CLIP_L_PATH="/root/autodl-tmp/fluxgym/models/clip/clip_l.safetensors"
export T5XXL_PATH="/root/autodl-tmp/fluxgym/models/clip/t5xxl_fp16.safetensors"
export VAE_PATH="/root/autodl-tmp/fluxgym/models/vae/ae.sft"
```

> China users can try my [AutoDL](https://www.autodl.com/) mirror [Fluxgym_Starter](https://www.codewithgpu.com/i/tj-scripts/fluxgym/Fluxgym_Starter).

## 2. Start via `python app.py`

Go back to the root `fluxgym` folder, with the venv activated, run:

```
python app.py
```

> Make sure to have the venv activated before running `python app.py`.
>
> Windows: `env/Scripts/activate`
> Linux: `source env/bin/activate`

## 3. Install via Docker

First clone Fluxgym and kohya-ss/sd-scripts:

```
git clone https://github.com/cocktailpeanut/fluxgym
cd fluxgym
git clone -b sd3 https://github.com/kohya-ss/sd-scripts
```
Check your `user id` and `group id` and change it if it's not 1000 via `environment variables` of `PUID` and `PGID`. 
You can find out what these are in linux by running the following command: `id`

Now build the image and run it via `docker-compose`:
```
docker compose up -d --build
```

Open web browser and goto the IP address of the computer/VM: http://localhost:7860

# Usage

The usage is pretty straightforward:

1. Enter the lora info
2. Upload images and caption them (using the trigger word)
3. Click "start".

That's all!

![flow.gif](flow.gif)

# Configuration

## Sample Images

By default fluxgym doesn't generate any sample images during training.

You can however configure Fluxgym to automatically generate sample images for every N steps. Here's what it looks like:

![sample.png](sample.png)

To turn this on, just set the two fields:

1. **Sample Image Prompts:** These prompts will be used to automatically generate images during training. If you want multiple, separate teach prompt with new line.
2. **Sample Image Every N Steps:** If your "Expected training steps" is 960 and your "Sample Image Every N Steps" is 100, the images will be generated at step 100, 200, 300, 400, 500, 600, 700, 800, 900, for EACH prompt.

![sample_fields.png](sample_fields.png)

## Advanced Sample Images

Thanks to the built-in syntax from [kohya/sd-scripts](https://github.com/kohya-ss/sd-scripts?tab=readme-ov-file#sample-image-generation-during-training), you can control exactly how the sample images are generated during the training phase:

Let's say the trigger word is **hrld person.** Normally you would try sample prompts like:

```
hrld person is riding a bike
hrld person is a body builder
hrld person is a rock star
```

But for every prompt you can include **advanced flags** to fully control the image generation process. For example, the `--d` flag lets you specify the SEED.

Specifying a seed means every sample image will use that exact seed, which means you can literally see the LoRA evolve. Here's an example usage:

```
hrld person is riding a bike --d 42
hrld person is a body builder --d 42
hrld person is a rock star --d 42
```

Here's what it looks like in the UI:

![flags.png](flags.png)

And here are the results:

![seed.gif](seed.gif)

In addition to the `--d` flag, here are other flags you can use:


- `--n`: Negative prompt up to the next option.
- `--w`: Specifies the width of the generated image.
- `--h`: Specifies the height of the generated image.
- `--d`: Specifies the seed of the generated image.
- `--l`: Specifies the CFG scale of the generated image.
- `--s`: Specifies the number of steps in the generation.

The prompt weighting such as `( )` and `[ ]` also work. (Learn more about [Attention/Emphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis))

## Publishing to Huggingface

1. Get your Huggingface Token from https://huggingface.co/settings/tokens
2. Enter the token in the "Huggingface Token" field and click "Login". This will save the token text in a local file named `HF_TOKEN` (All local and private).
3. Once you're logged in, you will be able to select a trained LoRA from the dropdown, edit the name if you want, and publish to Huggingface.

![publish_to_hf.png](publish_to_hf.png)


## Advanced

The advanced tab is automatically constructed by parsing the launch flags available to the latest version of [kohya sd-scripts](https://github.com/kohya-ss/sd-scripts). This means Fluxgym is a full fledged UI for using the Kohya script.

> By default the advanced tab is hidden. You can click the "advanced" accordion to expand it.

![advanced.png](advanced.png)


## Advanced Features

### Uploading Caption Files

You can also upload the caption files along with the image files. You just need to follow the convention:

1. Every caption file must be a `.txt` file.
2. Each caption file needs to have a corresponding image file that has the same name.
3. For example, if you have an image file named `img0.png`, the corresponding caption file must be `img0.txt`.
