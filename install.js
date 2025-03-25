module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "git config --global --add safe.directory '*'",
          "git clone -b sd3 https://github.com/kohya-ss/sd-scripts"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        path: "sd-scripts",
        venv: "../env",
        message: [
          "uv pip install -r requirements.txt",
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip uninstall -y diffusers[torch] torch torchaudio torchvision",
          "uv pip install -r requirements.txt",
          "uv pip install -U bitsandbytes"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          // xformers: true   // uncomment this line if your project requires xformers
        }
      }
    },
    {
      method: "fs.link",
      params: {
        drive: {
          vae: "models/vae",
          clip: "models/clip",
          unet: "models/unet",
          loras: "outputs",
        },
        peers: [
          "https://github.com/pinokiofactory/stable-diffusion-webui-forge.git",
          "https://github.com/pinokiofactory/comfy.git",
          "https://github.com/cocktailpeanutlabs/comfyui.git",
          "https://github.com/cocktailpeanutlabs/fooocus.git",
          "https://github.com/cocktailpeanutlabs/automatic1111.git",
        ]
      }
    },
//    {
//      method: "fs.download",
//      params: {
//        uri: [
//          "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true",
//          "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors?download=true",
//        ],
//        dir: "models/clip"
//      }
//    },
//    {
//      method: "fs.download",
//      params: {
//        uri: [
//          "https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/ae.sft?download=true",
//        ],
//        dir: "models/vae"
//      }
//    },
//    {
//      method: "fs.download",
//      params: {
//        uri: [
//          "https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/flux1-dev.sft?download=true",
//        ],
//        dir: "models/unet"
//      }
//    },
    {
      method: "fs.link",
      params: {
        venv: "env"
      }
    }
  ]
}
