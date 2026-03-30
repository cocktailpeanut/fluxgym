module.exports = {
  requires: {
    bundle: "ai",
  },
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
          "uv pip uninstall diffusers[torch] torch",
          "uv pip install -r requirements.txt",
          "uv pip install -U bitsandbytes hf-xet"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          // xformers: true
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
          "https://github.com/pinokiofactory/MagicQuill.git",
          "https://github.com/cocktailpeanutlabs/comfyui.git",
          "https://github.com/cocktailpeanutlabs/fooocus.git",
          "https://github.com/cocktailpeanutlabs/automatic1111.git",
          "https://github.com/6Morpheus6/forge-neo.git"
        ]
      }
    }
  ]
}
