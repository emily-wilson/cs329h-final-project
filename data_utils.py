from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, Dataset
from typing import Union, Optional
from transformers.utils import PaddingStrategy
import soundfile as sf
import io
import random
import pandas as pd

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


def get_populated_dataset(num_contexts = 2):
    audio_dataset = load_dataset(
        "datapointai/text-to-speech-human-preferences-315k",
        "audio",
        split="train",
        token=True,
    )

    def get_audio_samples(sample):
        all_samples = sample["audio"].get_all_samples()
        sample["samples"] = all_samples.data

        return sample
    print("processing audio dataset")
    audio_dataset = audio_dataset.map(get_audio_samples, remove_columns=["audio", "prompt_id", "prompt_index", "category", "sample_rate", "channels", "bit_depth", 'total_samples', 'duration_seconds', 'byte_size', 'sha256', "model_id"])

    pairs = load_dataset(
            "datapointai/text-to-speech-human-preferences-315k",
            "pairs",
            split="train",
            token=True,
        )
    pairs = pairs.remove_columns(["category", 'votes_a', 'votes_b', 'label_a', 'label_b', 'trust_weighted_votes_a', 'trust_weighted_votes_b', 'winner', "num_votes", "prompt_id"])
    responses = load_dataset(
                "datapointai/text-to-speech-human-preferences-315k",
                "responses",
                split="train",
                token=True,
            )
    responses = responses.remove_columns(["category", 'trust_score', 'time_taken_ms', 'completed_at', 'country'])

    pd_audio = audio_dataset.to_pandas()
    print(pd_audio.columns)
    pd_pairs = pairs.to_pandas()
    print(pd_pairs.columns)
    pd_responses = responses.to_pandas()
    print(pd_responses.columns)

    pairs_with_audios = pd.merge(pd_pairs, pd_audio, "inner", left_on="audio_a_key", right_on="audio_key")
    print(pairs_with_audios.columns)
    pairs_with_audios = pd.merge(pairs_with_audios, pd_audio, "inner", left_on="audio_b_key", right_on="audio_key", suffixes=["_a", "_b"])
    print(pairs_with_audios.columns)
    pairs_with_audios = pairs_with_audios.drop(columns=["audio_a_key", "audio_b_key", "audio_key_a", "audio_key_b"])

    filled_responses = pd.merge(pd_responses, pairs_with_audios, "inner", on="pair_key")
    print(filled_responses.columns)
    filled_responses = filled_responses.drop(columns=["pair_key"])

    responses = Dataset.from_pandas(filled_responses, preserve_index=False)

    # responses_by_annotator = {}
    # for response in responses:
    #     annotator_id = response["annotator"]
    #     if annotator_id not in responses_by_annotator:
    #         responses_by_annotator[annotator_id] = []
    #     responses_by_annotator[annotator_id].append(response)
        # pair = pair_by_pair_id[response["pair_key"]]

    # def add_contexts(sample):
    #     annotator_samples = responses_by_annotator[sample["annotator_id"]]
    #     annotator_samples = annotator_samples.filter(lambda r: r["prompt_id"] != sample["prompt_id"])
    #     contexts = random.sample(annotator_samples, num_contexts)
    #     print(contexts)
    #     context_ids_chosen = [c["chosen_samples"] for c in contexts]
    #     context_ids_rejected = [c["rejected_samples"] for c in contexts]

    # responses = responses.map(add_contexts)
    return responses

if __name__ == "__main__":
    get_populated_dataset()


