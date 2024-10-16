from typing import List, Optional, Tuple
from collections import defaultdict

from transformers import PreTrainedTokenizerFast
from datasets import Dataset, Sequence, ClassLabel


def assign_tags_to_dataset(
        hf_dataset: Dataset,
        tag_key: str,
        new_tag_key: str,
        iob_tags: bool = True,
        tag_strings: Optional[List[str]] = None
) -> Tuple[Dataset, list]:
    """
    Method assigns label ids to dataset for training or prediction purposes.

    :param hf_dataset: dataset witch annotated tokens
    :param tag_key: column with current tags
    :param new_tag_key: column name for new column tag ids
    :param iob_tags: flag if tags follow the IOB (Inside-Outside-Beginning) format
    :param tag_strings: list of unique NER tags
    :return:
    """
    # get unique tags
    if tag_strings is None:
        tag_strings = get_unique_tags(hf_dataset['train'], tag_key)
        print(f"Found the following tags: {tag_strings}")

    features = hf_dataset['train'].features
    tags = []
    # append tag for non entities
    if "O" not in tag_strings:
        tags.append("O")
    
    for tag in tag_strings:
        if iob_tags:
            if tag != "O":
                tags.append("B-" + tag)
                tags.append("I-" + tag)
        else:
            tags.append(tag)
    # assign tag integer ids to labels
    tag2idx = defaultdict(int)
    tag2idx.update({t: i for i, t in enumerate(tags)})
    # assign tag Ids to dataset
    hf_dataset = hf_dataset.map(lambda e: {new_tag_key: [tag2idx[t] for t in e[tag_key]]})
    features[new_tag_key] = Sequence(ClassLabel(num_classes=len(tags), names=(tags)))
    hf_dataset = hf_dataset.cast(features)

    return hf_dataset, tags


def get_unique_tags(data: List[dict], tag_key: str) -> List[str]:
    """
    Method to extract all unique tags in dataset
    """
    unique_tags = set()
    for sent in data:
        for tag in sent[tag_key]:
            unique_tags.add(tag)
    
    return list(unique_tags)


def align_labels_with_tokens(labels: List[int], word_ids: List[Optional[int]], iob_flag: bool = True) -> List:
    """
    Method to handle special tokens ([CLS], [SEP]) from tokenizer and to extend labels to split tokens.
    example:
        base_token: 'Tabakkonsum' / label: 'Finding'
        tokenizer_tokens: ['Tabak', '##konsum'] / label: ['B-Finding', 'I-Finding']
    :param labels:
    :param word_ids:
    :param iob_flag: flag if tokens and tags are chunked in IOB (Inside-Outside-Beginning) format or not.
    :return:
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if iob_flag:
                if label % 2 == 1:
                    label += 1
            new_labels.append(label)

    return new_labels


def build_tokenize_and_align_labels_func(
        tokenizer: PreTrainedTokenizerFast,
        tag_key: str,
        token_key: str = "tokens",
        iob_flag: bool = True
):
    """
    Method to pass arguments to `tokenize_and_align_labels` method
    """
    def tokenize_and_align_labels(examples: dict):
        """
        Method to tokenzie list of tokens and assign labels to tokens
        """
        tokenized_inputs = tokenizer(
            examples[token_key], truncation=True, is_split_into_words=True
        )
        all_labels = examples[tag_key]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids, iob_flag))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    return tokenize_and_align_labels
