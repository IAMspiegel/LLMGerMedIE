from typing import List, Optional, Tuple
from os import path

from datasets import load_dataset, load_metric, Sequence, ClassLabel, DatasetDict, Dataset
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments, pipeline, DataCollatorForTokenClassification
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
#import error_analysis
import pandas as pd

from nervaluate import Evaluator

metric = load_metric("seqeval")


def ggponc_instruction_prompt(sample: dict) -> str:
    return f"""### Instruction:
Extrahiere die Medikamente, Behandlung und Diagnosen aus dem folgendem Text. Gib die gefundenen Entitäten als Liste aus.

### Input:
{sample['input']}

### Response:
"""

def evaluate_ggponc_prediction(test_data: Dataset, pred_data: dict) -> Tuple[dict, dict]:
    # build dictionary
    pred_dict = {e["id"]: e for e in pred_data['predictions']}

    # create dataset with true and pred token
    data_x_y = []
    for data_record in test_data:
        _tokens = []
        sent_id = data_record['fname'] + str(data_record['sentence_id'])
        for token in data_record['tokens']:
            # check for token if model assigned entity
            _tokens.append(find_token_ent(token, sent_id, pred_dict))
        data_record['pred_tokens'] = _tokens
        data_x_y.append(data_record)
    
    true_tags: List[List[str]] = []
    pred_tags: List[List[str]] = []

    for record in data_x_y:
        true_tags.append(record['tags'])
        pred_tags.append(record['pred_tokens'])

    evaluator = Evaluator(true=true_tags, pred=pred_tags, tags=['Clinical_Drug', 'Diagnosis_or_Pathology', 'Therapeutic'], loader='list')
    results, results_per_tag = evaluator.evaluate()

    return results, results_per_tag


def build_xy_data(predictions, test_data, map_dict, combine_treats=False):
    """Method to create list of true and predicted tags to feed into `nervaluate.Evaluator`"""

    # get prediction for annotated token
    data_x_y = []
    for i, data_record in enumerate(test_data):
        pred_label = []
        for pred in predictions[i]:
            if map_dict.get(pred):
                pred_label.append(map_dict[pred])
            else:
                pred_label.append("O")
        assert len(pred_label) == len(data_record['tags'])
        data_record['pred_tags'] = pred_label
        data_x_y.append(data_record)


    true_tags = []
    pred_tags = []
    for record in data_x_y:
        if combine_treats:
            rtags = [t.replace('Diagnostic', 'Therapeutic') for t in record['tags']]
            true_tags.append(rtags)
        else:
            true_tags.append(record['tags'])
        pred_tags.append(record['pred_tags'])
    
    return true_tags, pred_tags


def clean_data(dataset: Dataset) -> Dataset:
    """Remove private chars"""
    new_dataset = []
    for r_idx, record in enumerate(dataset):
        pop_indx = []
        # loof for illegal char
        for i, token in enumerate(record['tokens']):
            if token in ['\uf0b3', '\xad', '\u00ad']:
                pop_indx.append(i)
        # remove chars
        if pop_indx:
            for idx in reversed(pop_indx):
                print(r_idx)
                record['tokens'].pop(idx)
                record['tags'].pop(idx)
        new_dataset.append(record)
    return new_dataset


def _assign_entity(tok: str, ent_list: list, ent_type: str) -> Optional[str]:
    # ignore 'Diagnosen:'
    for ent in ent_list:
        if 'Diagnosen' not in ent:
            if ' ' in ent:
                for i, e_split in enumerate(ent.split()):
                    if i == 0 and e_split.lower() == tok:
                        return "B-" + ent_type
                    elif e_split.lower() == tok:
                        return "I-" + ent_type
            else:
                if ent.lower() == tok:
                    return "B-" + ent_type
    return None


def find_token_ent(token: str, sent_id: str, pred_data: dict) -> str:
    pred_data_dict = pred_data[sent_id]
    for e_key, e_type in [('Medikamente', 'Clinical_Drug'), ('Diagnose', 'Diagnosis_or_Pathology'), ('Behandlung', 'Therapeutic')]:

        if pred_data_dict['parsed_output'].get(e_key):
            _ent = _assign_entity(token.lower(), pred_data_dict['parsed_output'][e_key], e_type)
            if _ent:
                return _ent
    return "O"


def load_ggponc_dataset(base_dir: str, anno_level: str, anno_span: str, data_types: List[str]):
    # load hf data
    data_files_dict = build_data_file_paths(base_dir, anno_level, anno_span, data_types)
    ggponc_dataset = load_dataset('json', data_files=data_files_dict)
    # build tags
    tag_strings = ['Other_Finding', 'Diagnosis_or_Pathology', 'Therapeutic', 'Diagnostic', 'Nutrient_or_Body_Substance', 'External_Substance', 'Clinical_Drug' ]
    features = ggponc_dataset["train"].features
    tags = []
    tags.append("O")
    for tag in tag_strings:
        tags.append("B-" + tag)
        tags.append("I-" + tag)
    # assign tag integer ids to labels
    tag2idx = defaultdict(int)
    tag2idx.update({t: i for i, t in enumerate(tags)})
    # assign tag Ids to dataset
    ggponc_dataset = ggponc_dataset.map(lambda e: {"_tags" : [tag2idx[t] for t in e["tags"]]})
    features["_tags"] = Sequence(ClassLabel(num_classes=len(tags), names=(tags)))
    ggponc_dataset = ggponc_dataset.cast(features)

    return ggponc_dataset, tags


def build_data_file_paths(base_dir: str, anno_level: str, anno_span: str, data_type: List[str] = ['train', 'test', 'dev']) -> dict:
    file_dict = {
        k: path.join(base_dir, anno_level, anno_span, f"{k}_{anno_level}_{anno_span}.json") for k in data_type
    } 
    return file_dict


class LabelAligner():
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_and_align_labels(self, examples, label_all_tokens=True):
        tokenized_inputs = self.tokenizer(examples["tokens"], 
                                          truncation=True, 
                                          is_split_into_words=True, 
                                          return_offsets_mapping=True,
                                          return_special_tokens_mask=True)

        labels = []
        for i, label in enumerate(examples["_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

def load_custom_dataset(train, dev, test, tag_strings=None):
    tags = []
    dataset = load_dataset("json", data_files={'train' : str(train), 'dev' : str(dev), 'test' : str(test)})
    features = dataset["train"].features

    tags.append("O")
    for tag in tag_strings:
        tags.append("B-" + tag)
        tags.append("I-" + tag)
    tag2idx = defaultdict(int)
    tag2idx.update({t: i for i, t in enumerate(tags)})
    dataset = dataset.map(lambda e: {"_tags" : [tag2idx[t] for t in e["tags"]]})
    features["_tags"] = Sequence(ClassLabel(num_classes=len(tags), names=(tags)))
        
    dataset = dataset.cast(features)
        
    return dataset, tags

def compute_metrics(label_list, entity_level_metrics):
    def _compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        if entity_level_metrics:
            final_results = {}
            # Unpack nested dictionaries
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return _compute_metrics
