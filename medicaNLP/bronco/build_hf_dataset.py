from os import path, listdir
from typing import List

from datasets import Dataset


def build_dataset_from_dir(file_dir: str,) -> Dataset:
    """
    Method to build a hugginface dataset from Bronnco150 conll files
    """
    hf_data: List[dict] = []

    for file in listdir(file_dir):
        hf_data += file_to_hf_json_format(path.join(file_dir, file))

    def dataset_generator():
        for data_dict in hf_data:
            yield data_dict

    hf_dataset = Dataset.from_generator(dataset_generator)

    return hf_dataset


def file_to_hf_json_format(file_path: str) -> List[dict]:
    # helper vars to store data
    sentence_dicts: List[dict] = []
    sentence_token: List[str] = []
    sentence_tag: List[str] = []
    sentence_id: int = 1

    with open(file_path, "r") as fh:
        for line in fh:
            # indicates new sentence
            if line == "\n":
                # store collected data
                sentence_dicts.append(
                    {
                        "fname": path.basename(file_path),
                        "sentence_id": sentence_id,
                        "tokens": sentence_token.copy(),
                        "tags": sentence_tag.copy()
                    }
                )
                # clear lists and increase sentence id
                sentence_token.clear()
                sentence_tag.clear()
                sentence_id += 1

            else:
                # append token and tag to sentence lists
                token, _pos, tag = line.replace('\n', '').split('\t', maxsplit=3)
                sentence_token.append(token)
                sentence_tag.append(tag)

    return sentence_dicts
