from pathlib import Path
import modal

## Most of this comes from the modal setup instructions: https://modal.com/docs/guide
app = modal.App("vpl")
image = modal.Image.debian_slim().pip_install_from_requirements("./requirements.txt")


## This will train the reward function following the VPL algorithm
@app.function(gpu="a100", image=image)
def train_reward_model():
    ## Training code comes largely from HuggingFace docs (https://huggingface.co/docs/transformers/en/tasks/text-to-speech)
    ## some training code implementation is also from https://github.com/WEIRDLabUW/vpl_llm. This is the implementation of the VPL paper cited in the report.
    
    
    pass

@app.function(gpu="a100", image=image)
def run_ppo():
    ## This also follows the HuggingFace docs (https://huggingface.co/docs/transformers/en/tasks/text-to-speech), VPL paper is not applicable here
    
    pass