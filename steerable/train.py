from pathlib import Path

from transformers import TrainingArguments
from steerable_utils import SteerablePluralismModel, SteerablePluralismTrainer
from qwen_tts import Qwen3TTSTokenizer
from datasets import load_dataset
from data_utils import get_dataset_with_context

import modal
import wandb

## Most of this comes from the modal setup instructions: https://modal.com/docs/guide
app = modal.App("steerable-pluralism")
image = modal.Image.debian_slim().pip_install_from_requirements("./requirements.txt")

@app.function(gpu="a100", image=image)
def run_training():
    wandb.init()
    model = SteerablePluralismModel()
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
    )

    dataset = get_populated_dataset()

    args = TrainingArguments(
        # other args and kwargs here
        report_to="wandb",  # enable logging to W&B
        run_name="initial run",  # name of the W&B run (optional)
        logging_steps=1,  # how often to log to W&B
    )

    trainer = SteerablePluralismTrainer(
        model=model,
        args=args
    )

    trainer.train()