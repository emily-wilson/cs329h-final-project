from pathlib import Path

from transformers import TrainingArguments
from steerable.steerable_utils import SteerablePluralismModel, SteerablePluralismTrainer
from qwen_tts import Qwen3TTSTokenizer
from datasets import load_dataset
from utils.data_utils import get_dataset_with_context, DataCollator

import modal
import wandb

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

## Most of this comes from the modal setup instructions: https://modal.com/docs/guide
app = modal.App("steerable-pluralism")
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .add_local_dir(ROOT / "steerable", remote_path="/root/steerable")
    .add_local_dir(ROOT / "utils", remote_path="/root/utils")
)

@app.function(gpu="t4", image=image, secrets=[modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface-secret")]) 
def run_training():
    wandb.init()
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
    )

    dataset = get_dataset_with_context()
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    steerable_model = SteerablePluralismModel(tokenizer=tokenizer, input_dim=5000, hidden_dim=100)

    args = TrainingArguments(
        # other args and kwargs here
        report_to="wandb",  # enable logging to W&B
        run_name="initial run",  # name of the W&B run (optional)
        logging_steps=1,  # how often to log to W&B
        remove_unused_columns=False,
        num_train_epochs=2
    )

    trainer = SteerablePluralismTrainer(
        model=steerable_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(
            tokenizer=tokenizer,
            max_length=5000
        )
    )

    trainer.train()

if __name__ == "__main__":
    run_training()