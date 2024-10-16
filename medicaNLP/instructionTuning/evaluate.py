import json
from typing import Optional, List, Tuple

import pandas as pd

from datasets import load_dataset
from nervaluate import Evaluator

from medicalNLP.bronco.bronco_utils import find_token_ent


def evaluate_cardio_prediction_map(test_path: str, prediction_path: str) -> Tuple[dict, dict]:
    """
    Method to evaluate LLM output on CARDIO:DE datasample
    DRUG entity is mapped to DRUG + ACTIVEING
    """
    test_data = load_dataset('json', data_files={'test': test_path})
    # load prediction
    with open(prediction_path, 'r') as fh:
        pred_test = json.load(fh)
    
    # build dictionary
    pred_dict = {e["id"]: e for e in pred_test['prediction']}

    # create dataset with true and pred token

    data_x_y = []
    for data_record in test_data['test']:
        _tokens = []
        sent_id = data_record['fname']+str(data_record['sentence_id'])
        for token in data_record['tokens']:
            # check for token if model assigned entity
            _tokens.append(find_token_ent(token, sent_id, pred_dict))
        data_record['pred_tokens'] = _tokens
        data_x_y.append(data_record)
    

    true_tags: List[List[str]] = []
    pred_tags: List[List[str]] = []

    for record in data_x_y:
        true_tags.append([t.replace('ACTIVEING', 'DRUG') for t in record['tags']])  #change to DRUG
        pred_tags.append(record['pred_tokens']) 


    evaluator = Evaluator(true=true_tags, pred=pred_tags, tags=['DRUG'], loader='list')
    results, results_per_tag = evaluator.evaluate()

    return results, results_per_tag


def evaluate_bronco_prediction(test_path: str, pred_data: dict) -> Tuple[dict, dict]:
    test_data = load_dataset('json', data_files={'test': test_path})
    
    # build dictionary
    pred_dict = {e["id"]: e for e in pred_data['prediction']}

    # create dataset with true and pred token
    data_x_y = []
    for data_record in test_data['test']:
        _tokens = []
        sent_id = data_record['id']
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

    evaluator = Evaluator(true=true_tags, pred=pred_tags, tags=['MED', 'DIAG', 'TREAT'], loader='list')
    results, results_per_tag = evaluator.evaluate()

    return results, results_per_tag