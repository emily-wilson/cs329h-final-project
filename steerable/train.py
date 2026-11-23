from pathlib import Path
from steerable_utils import SteerablePluralismModel, SteerablePluralismTrainer
from qwen_tts import Qwen3TTSTokenizer
from datasets import load_dataset
from data_utils import get_populated_dataset

import modal

## Most of this comes from the modal setup instructions: https://modal.com/docs/guide
app = modal.App("steerable-pluralism")
image = modal.Image.debian_slim().pip_install_from_requirements("./requirements.txt")

@app.function(gpu="a100", image=image)
def run_training():
    model = SteerablePluralismModel()
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
    )

    dataset = get_populated_dataset()

    trainer = SteerablePluralismTrainer(
        model=model
    )

    trainer.train()