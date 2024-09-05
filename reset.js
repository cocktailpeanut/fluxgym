module.exports = {
  run: [{
    method: "fs.rm",
    params: {
      path: "sd-scripts"
    }
  }, {
    method: "fs.rm",
    params: {
      path: "env"
    }
  }]
}
