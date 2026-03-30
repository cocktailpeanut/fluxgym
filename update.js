module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      path: "sd-scripts",
      message: "git pull"
    }
  }, {
    method: "fs.rm",
    params: {
      path: "env"
    }
  }, {
    method: "shell.run",
    params: {
      path: "sd-scripts",
      venv: "../env",
      message: [
        "uv pip install -r requirements.txt",
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      message: [
        "pip uninstall -y diffusers[torch] torch torchaudio torchvision",
        "uv pip install -r requirements.txt",
      ]
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
        // xformers: true   // uncomment this line if your project requires xformers
      }
    }
  }, {
    method: "fs.link",
    params: {
      venv: "env"
    }
  }]
}
