from pathlib import Path

import modal

## Most of this comes from the modal setup instructions: https://modal.com/docs/guide
app = modal.App("benchmarking")
image = modal.Image.debian_slim().pip_install_from_requirements("./requirements.txt")

@app.function(gpu="a100", image=image)
def run_benchmarking():
    pass