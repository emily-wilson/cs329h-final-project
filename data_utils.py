from transformers import PreTrainedTokenizerBase
from datasets import load_dataset
from typing import Union, Optional
from transformers.utils import PaddingStrategy
import random

def prefix_dict_items(obj: dict, prefix: str):
    result = {}

    for key in obj.keys():
        result[f"{prefix}_{key}"] = obj[key]

    return result

class DataCollator():
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch = {}

        audios_chosen = []
        audios_rejected = []
        for feature in features:
            audios_chosen.append(feature["samples"].data.flatten())
            audios_rejected.append(feature["samples"].data.flatten())

        input_ids = tokenizer(audios_chosen + audios_rejected, padding=True, return_tensors="pt")
        tensor_len = input_ids.shape[1]
        input_ids = input_ids.reshape((2, len(features), tensor_len))

        return {
            "chosen_input_ids": input_ids[0],
            "rejected_input_ids": input_ids[1]
        }

def get_dataset_with_context(num_contexts = 2):
    responses = load_dataset(
            "emilymcf/processed_tts-human_preferences",
            split="train",
            token=True,
        )
    
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

        audio_contexts_chosen = [s["chosen_audio"] for s in sampled_responses]
        audio_contexts_rejected = [s["rejected_audio"] for s in sampled_responses]

        return sample | {
            "audio_contexts_chosen": audio_contexts_chosen,
            "audio_contexts_rejected": audio_contexts_rejected
        }
    responses = responses.map(fill_context)
    return responses

if __name__ == "__main__":
    get_dataset_with_context()


