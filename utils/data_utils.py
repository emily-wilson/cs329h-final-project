from transformers import PreTrainedTokenizerBase
from datasets import load_dataset
from typing import Union, Optional
from transformers.utils import PaddingStrategy
import random
from dataclasses import dataclass
import torch
import json
from torch.nn.utils.rnn import pad_sequence

def prefix_dict_items(obj: dict, prefix: str):
    result = {}

    for key in obj.keys():
        result[f"{prefix}_{key}"] = obj[key]

    return result

@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    split: str = "train"
    padding_value: int = -1

    def __get_user_context(self, audio_keys, audio_by_audio_key):
        user_contexts = []

        for context in audio_keys:
            context_audio = audio_by_audio_key[context]

            context_ids = context_audio.get_all_samples().data.flatten()
            user_contexts.append(context_ids)

        return user_contexts


    def __call__(self, features):
        batch = {}

        audio = load_dataset(
            "datapointai/text-to-speech-human-preferences-315k",
            "audio",
            split=self.split,
            token=True,
            columns=["audio_key", "audio"]
        )

        audio_by_audio_key = {a["audio_key"]: a["audio"] for a in audio}

        audios_chosen = []
        audios_rejected = []
        contexts_chosen = []
        contexts_rejected = []
        for feature in features:
            chosen_audio = audio_by_audio_key[feature["chosen_audio_key"]]
            rejected_audio = audio_by_audio_key[feature["rejected_audio_key"]]

            chosen_audio_ids = chosen_audio.get_all_samples().data.flatten()
            rejected_audio_ids = rejected_audio.get_all_samples().data.flatten()
            
            audios_chosen.append(chosen_audio_ids)
            audios_rejected.append(rejected_audio_ids)

            contexts_chosen.append(self.__get_user_context(feature["audio_keys_contexts_chosen"], audio_by_audio_key))
            contexts_rejected.append(self.__get_user_context(feature["audio_keys_contexts_rejected"], audio_by_audio_key))

        input_ids = pad_sequence(
            torch.cat(audios_chosen, audios_rejected),
            batch_first=True,
            padding_value=self.padding_value
        )
        tensor_len = input_ids.shape[1]
        input_ids = input_ids.reshape((2, len(features), tensor_len))

        context_ids = pad_sequence(
            torch.cat(contexts_chosen, contexts_rejected),
            batch_first=True,
            padding_value=self.padding_value
        )
        context_tensor_len = context_ids.shape[1]
        context_ids = context_ids.reshape((2, len(features), context_tensor_len))

        return {
            "chosen_input_ids": input_ids[0],
            "rejected_input_ids": input_ids[1],
            "context_ids_chosen": context_ids[0],
            "context_ids_rejected": context_ids[1]
        }

def get_dataset_with_context(num_contexts = 2, split="train"):
    pairs = load_dataset(
        "datapointai/text-to-speech-human-preferences-315k",
        "pairs",
        split=split,
        token=True,
        columns=["pair_key", "model_a", "model_b", "audio_a_key", "audio_b_key"]
    )

    pairs_by_pair_key = {p["pair_key"]: p for p in pairs}
    
    responses = load_dataset(
        "datapointai/text-to-speech-human-preferences-315k",
        "responses",
        split=split,
        token=True,
        columns=["pair_key", "prompt_id", "annotator", "chosen"]
    )

    def fill_responses(sample):
        pair = pairs_by_pair_key[sample["pair_key"]]
        if not pair:
            print("missing pair")
            return

        audio_a_key = pair["audio_a_key"]
        audio_b_key = pair["audio_b_key"]

        chosen_audio_key = audio_a_key if sample["chosen"] == 'a' else audio_b_key
        rejected_audio_key = audio_b_key if sample["chosen"] == 'a' else audio_a_key

        return {
            "prompt_id": sample["prompt_id"],
            "annotator": sample["annotator"],
            "chosen_model": pair["model_a"] if sample["chosen"] == 'a' else pair["model_b"],
            "rejected_model": pair["model_b"] if sample["chosen"] == 'a' else pair["model_a"],
            "chosen_audio_key": chosen_audio_key,
            "rejected_audio_key": rejected_audio_key,
            "audio_a_key": audio_a_key,
            "audio_b_key": audio_b_key
        }

    responses = responses.map(fill_responses)

    responses_by_annotator_id = {}
    for response in responses:
        if response["annotator"] not in responses_by_annotator_id:
            responses_by_annotator_id[response["annotator"]] = []
        responses_by_annotator_id[response["annotator"]].append(response)

    def fill_context(sample):
        all_responses = responses_by_annotator_id[sample["annotator"]]

        if len(all_responses) < num_contexts:
            return None
        
        sampled_responses = random.sample(all_responses, num_contexts)

        audio_contexts_chosen = [s["chosen_audio_key"] for s in sampled_responses]
        audio_contexts_rejected = [s["rejected_audio_key"] for s in sampled_responses]

        return sample | {
            "audio_keys_contexts_chosen": audio_contexts_chosen,
            "audio_keys_contexts_rejected": audio_contexts_rejected
        }
    responses = responses.map(fill_context)

    return responses

if __name__ == "__main__":
    dataset = get_dataset_with_context()

    audio = load_dataset(
        "datapointai/text-to-speech-human-preferences-315k",
        "audio",
        split="train",
        token=True,
        columns=["audio_key", "audio"]
    )

    audio_by_audio_key = {a["audio_key"]: a["audio"] for a in audio}

    def __get_user_context(audio_keys, audio_by_audio_key):
        user_contexts = []

        for context in audio_keys:
            context_audio = audio_by_audio_key[context]

            context_ids = context_audio.get_all_samples().data.flatten()
            user_contexts.append(context_ids)

        return user_contexts

    audios_chosen = []
    audios_rejected = []
    contexts_chosen = []
    contexts_rejected = []
    for feature in dataset:
        chosen_audio = audio_by_audio_key[feature["chosen_audio_key"]]
        rejected_audio = audio_by_audio_key[feature["rejected_audio_key"]]

        chosen_audio_ids = chosen_audio.get_all_samples().data.flatten()
        rejected_audio_ids = rejected_audio.get_all_samples().data.flatten()
        
        audios_chosen.append(chosen_audio_ids)
        audios_rejected.append(rejected_audio_ids)

        contexts_chosen.append(__get_user_context(feature["audio_keys_contexts_chosen"], audio_by_audio_key))
        contexts_rejected.append(__get_user_context(feature["audio_keys_contexts_rejected"], audio_by_audio_key))

    input_ids = pad_sequence(
        torch.cat([audios_chosen, audios_rejected], dim=0),
        batch_first=True,
        padding_value=-1
    )
    tensor_len = input_ids.shape[1]
    input_ids = input_ids.reshape((2, len(features), tensor_len))

    context_ids = pad_sequence(
        torch.cat([contexts_chosen, contexts_rejected]),
        batch_first=True,
        padding_value=-1
    )
    context_tensor_len = context_ids.shape[1]
    context_ids = context_ids.reshape((2, len(features), context_tensor_len))
