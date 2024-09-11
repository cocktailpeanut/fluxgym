# Flux Gym

Dead simple web UI for training FLUX LoRA **with LOW VRAM (12GB/16GB/20GB) support.**

- **Frontend:** The WebUI forked from [AI-Toolkit](https://github.com/ostris/ai-toolkit) (Gradio UI created by https://x.com/multimodalart)
- **Backend:** The Training script powered by [Kohya Scripts](https://github.com/kohya-ss/sd-scripts)

![screenshot.png](screenshot.png)

# How people are using Fluxgym

- https://x.com/cocktailpeanut/status/1833881392482066638
- https://x.com/Alone1Moon/status/1833348850662445369
- https://x.com/_f_ai_9/status/1833485349995397167
- https://x.com/intocryptoast/status/1833061082862412186
- https://x.com/cocktailpeanut/status/1833888423716827321
- https://x.com/cocktailpeanut/status/1833884852992516596
- https://x.com/cocktailpeanut/status/1833885335077417046
- https://x.com/NiwonArt/status/1833565746624139650
- https://x.com/cocktailpeanut/status/1833884361986380117
- https://x.com/NiwonArt/status/1833599399764889685
- https://x.com/LikeToasters/status/1832934391217045913
- https://x.com/cocktailpeanut/status/1832924887456817415
- https://x.com/cocktailpeanut/status/1832927154536902897
- https://x.com/YabaiHamster/status/1832697724690386992
- https://x.com/cocktailpeanut/status/1832747889497366706
- https://x.com/PhotogenicWeekE/status/1832720544959185202
- https://x.com/zuzaritt/status/1832748542164652390
- https://x.com/foxyy4i/status/1832764883710185880
- https://x.com/waynedahlberg/status/1832226132999213095
- https://x.com/PhotoGarrido/status/1832214644515041770
- https://x.com/cocktailpeanut/status/1832787205774786710
- https://x.com/cocktailpeanut/status/1832151307198541961
- https://x.com/cocktailpeanut/status/1832145996014612735
- https://x.com/cocktailpeanut/status/1832084951115972653
- https://x.com/cocktailpeanut/status/1832091112086843684

# What is this?

1. I wanted a super simple UI for training Flux LoRAs
2. The [AI-Toolkit](https://github.com/ostris/ai-toolkit) project is great, and the gradio UI contribution by [@multimodalart](https://x.com/multimodalart) is perfect, but the project only works for 24GB VRAM.
3. [Kohya Scripts](https://github.com/kohya-ss/sd-scripts) are very flexible and powerful for training FLUX, but you need to run in terminal.
4. What if you could have the simplicity of AI-Toolkit WebUI and the flexibility of Kohya Scripts?
5. Flux Gym was born. Supports 12GB, 16GB, 20GB VRAMs, and extensible since it uses Kohya Scripts underneath.

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
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

Now let's download the model checkpoints.

First, download the following models under the `models/clip` foder:

- https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true
- https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors?download=true

Second, download the following model under the `models/vae` folder:

- https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/ae.sft?download=true

Finally, donwload the following model under the `models/unet` folder:

- https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/flux1-dev.sft?download=true

The result file structure will be something like:

```
/models
  /clip
    clip_l.safetensors
    t5xxl_fp16.safetensors
  /unet
    flux1-dev.sft
  /vae
    ae.sft
/sd-scripts
/outputs
/env
app.py
requirements.txt
...
```

# Start

Go back to the root `fluxgym` folder, with the venv activated, run:

```
python app.py
```

> Make sure to have the venv activated before running `python app.py`.
>
> Windows: `env/Scripts/activate`
> Linux: `source env/bin/activate`

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
