import os
import time
import json
from typing import Tuple, List
from sklearn.model_selection import train_test_split

import torch
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


from medicaNLP.bronco.bronco_utils import parse_output
from medicaNLP.ggponc.hf_utils import _assign_entity
from medicaNLP.instructionTuning.build_instruct_data import build_gptnermed_instruct_data, build_instruction_dataset, I2B2_2010_INSTRUCTION


DATASET_OUTPUT_ENTITIES = {
    'gptnermed': ['Medikamente', 'Diagnose'],
    'gernermed': ['Medikamente', 'Diagnose'],
    'n2c2': ['Drug', 'Reason', 'ADE'],
    'i2b2': ['problem', 'treatment']
}

DATASET_EVAL_ENTITIES = {
    'gptnermed': ['Medikamente', 'Diagnose'],
    'gernermed': ['Drug'],
    'n2c2': ['Drug', 'Reason', 'ADE'],
    'i2b2': ['problem', 'treatment']
}


def instruction_prompt(sample: dict) -> str:
    return f"""### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n"""


def run_prediction(model_id: str, dataset_name: str):

    model_name = os.path.basename(model_id)


    if dataset_name == 'gptnermed':
        test_data = get_gptnermed_test()
        entities = ['Medikamente', 'Diagnose']
    elif dataset_name == 'gernermed':
        test_data = get_gernermed_test()
        entities = ['Medikamente']
    elif dataset_name == 'n2c2':
        test_data = get_n2c2_test()
        entities = ['Drug', 'Reason', 'ADE']
    elif dataset_name == 'i2b2':
        test_data = get_i2b2_2010_test()
        entities = ['problem', 'treatment']
    else:
        raise ValueError(f'Unknown dataset \'{dataset_name}\'. Abort.')

    # init model and tokenizer
    llm, tokenizer = get_model_and_tokenizer(model_id, use_flash=False)

    start = time.time()
    pred_test_data = predict_data(llm, tokenizer, test_data, entities)
    end = time.time()

    pred_datadict = {
        "model": model_name,
        "prediction": pred_test_data,
        "time": str(end - start)
    }

    with open(f'pred_{model_name}_{dataset_name}_test.json', "w") as fp:
        json.dump(pred_datadict, fp)

    print(f'Prediction took {end - start} sec')


def get_i2b2_2010_test():
    # i2b2 2010 data
    with open('/data/German-medical-texts/i2b2_2010/instruction_dataset.json', 'r') as fh:
        i2b2_beth = json.load(fh)
    with open('/data/German-medical-texts/i2b2_2010/partner_instruction_dataset.json', 'r') as fh:
        i2b2_partner = json.load(fh)

    # add instruction
    i2b2_data = i2b2_beth + i2b2_partner
    for record in i2b2_data:
        record['instruction'] = I2B2_2010_INSTRUCTION

    i2b2_train, i2b2_valid = train_test_split(i2b2_data, train_size=0.9, random_state=16)
    return i2b2_valid

def get_n2c2_test():
    # n2c2 2018 data
    with open('/data/German-medical-texts/n2c2_2018/instruction_dataset.json', 'r') as fh:
        n2c2_2018_train = json.load(fh)

    # join input lines
    for record in n2c2_2018_train:
        record['input'] = ''.join(record['input'])

    n2c2_2018_train, n2c2_2018_valid = train_test_split(n2c2_2018_train, train_size=0.9, random_state=16)
    return n2c2_2018_valid


def get_gernermed_test():
    # GerNerMed data
    gernermed_instruct_data = build_instruction_dataset(file_dict={
        "gernermed": "/data/German-medical-texts/GerNerMed/GERNERMED_dataset.json"
    })
    _gernermed_train, _gernermed_valid, gernermed_test = split_gernermed(gernermed_instruct_data)
    return gernermed_test


def get_gptnermed_test():
    # GPTNerMed data
    gptner_hfdata = load_dataset("jfrei/GPTNERMED")
    test_instruct = build_gptnermed_instruct_data(gptner_hfdata['test'])
    return test_instruct


def get_model_and_tokenizer(model_path: str, use_flash: bool) -> Tuple[AutoPeftModelForCausalLM, AutoTokenizer]:
    # init LM model and Tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        use_flash_attention_2=use_flash,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict_data(llm: AutoPeftModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, entities: List[str]) -> List[dict]:
    # var to store parsed output
    parsed_predictions: List[dict] = []

    count = 1
    data_len = len(dataset)

    for i, record in enumerate(dataset):

        print(f"{count}/{data_len}")
        count += 1

        prompt = instruction_prompt(record)

        # tokenize, predict and decode
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        outputs = llm.generate(
            input_ids=input_ids,
            max_new_tokens=240,
            do_sample=True,
            top_p=0.95,
            temperature=0.1
        )
        output_str = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]

        # append data
        if 'id' in record:
            _id = record['id']
        else:
            _id = i
        output_dict = {
            "id":_id,
            "raw_output": output_str
        }

        try:
            output_dict['parsed_output'] = parse_output(output_str, entities)
        except KeyboardInterrupt as keyint:
            raise keyint
        except SystemExit as sysexit:
            raise sysexit
        except:
            output_dict['parsed_output'] = ""

        parsed_predictions.append(output_dict)

    return parsed_predictions


def split_gernermed(dataset: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    train_data, test_data = train_test_split(dataset, train_size=0.9, random_state=16)
    train_data, valid_data = train_test_split(train_data, train_size=0.89, random_state=18)
    return train_data, valid_data, test_data


def evaluate_epochs(base_path: str, checkpoints: List[str], dataset_type: str, test_data: List[dict]) -> pd.DataFrame:
    """
    Method to evaluate test sample of instruction-tuning tran data
    """
    eval_results = []

    for epoch, chkpnt in enumerate(checkpoints):
        # read prediction
        with open(os.path.join(base_path, f"pred_checkpoint-{chkpnt}_{dataset_type}_test.json"), "r") as fh:
            pred_data = json.load(fh)

        # preproccess
        all_tokens = []
        true_tags = []
        pred_tags = []

        for i, record in enumerate(test_data):

            if dataset_type == 'i2b2':
                tokens = []
                tags = []
                _pred_tags = []
                # sort concepts by line and token index
                concept_dict = build_concept_dict(record['concepts'])
                # process per text line
                for line_indx, line in enumerate(record['input'].split('\n')):
                    indx = line_indx + record['start']
                    for tok_indx, token in enumerate(line.split()):
                        # get label for token
                        if concept_dict.get(indx):
                            tok_tag = concept_dict[indx][tok_indx]['label'] if concept_dict[indx].get(tok_indx) else 'O'
                        else:
                            tok_tag = 'O'
                        # append results
                        _pred_tags.append(
                            find_token_ent(token, pred_data['prediction'][i], DATASET_OUTPUT_ENTITIES[dataset_type])
                        )
                        tokens.append(token)
                        tags.append(tok_tag)
            else:
                # create token list and align tags to token
                tokens, tags = extract_tokens_and_labels(record, dataset_type)
                
                # get predicted entity for each token
                _pred_tags = [find_token_ent(t, pred_data['prediction'][i], DATASET_OUTPUT_ENTITIES[dataset_type]) for t in tokens]
                if dataset_type == 'gernermed':
                    # replace medikamente with drug
                    _pred_tags = [pt.replace('Medikamente', 'Drug') for pt in _pred_tags]

            all_tokens.append(tokens)
            true_tags.append(tags)
            pred_tags.append(_pred_tags)
        # evaluate
        evaluator = Evaluator(true=true_tags, pred=pred_tags, tags=DATASET_EVAL_ENTITIES[dataset_type], loader='list')
        results, results_per_tag = evaluator.evaluate()

        # store
        _result_dict = {
            'epoch': epoch + 1,
            'f1_enttype': results['ent_type']['f1'],
            'f1_strict': results['strict']['f1'],
        }
        for ent_type in DATASET_EVAL_ENTITIES[dataset_type]:
            _result_dict[f"f1_enttype_{ent_type}"] = results_per_tag[ent_type]['ent_type']['f1']
            _result_dict[f"f1_strict_{ent_type}"] = results_per_tag[ent_type]['strict']['f1']

        eval_results.append(_result_dict)

    return pd.DataFrame(eval_results)
        

def extract_tokens_and_labels(input_dict: dict, dataset_type: str) -> Tuple[List[str], List[str]]:
    """
    Method to create token list and align tags to token
    """
    # GPTNERMED
    if dataset_type == 'gptnermed':
        class_mapping = {0: 'Medikamente', 1: 'Dosis', 2: 'Diagnose'}
        sentence = input_dict['sentence']
        input_dict['ner_labels']['ner_class'] = [class_mapping[c] for c in input_dict['ner_labels']['ner_class']]
        # Create a list to store the NER class for each character in the sentence
        char_labels = _insert_label_gptnermed(input_dict, ['O'] * len(sentence))

    # GERNERMED
    elif dataset_type == 'gernermed':
        sentence = input_dict['de']
        char_labels = _insert_label_gernermed(input_dict, ['O'] * len(sentence))

    # n2c2 2018
    elif dataset_type == 'n2c2':
        sentence = input_dict['input']
        char_labels = _insert_label_n2c2(input_dict, ['O'] * len(sentence))


    # Tokenize the sentence
    tokens = sentence.split() #re.findall(r'\w+|\S', sentence)

    # Create lists to hold the final tokens and their corresponding NER classes
    final_tokens = []
    final_labels = []

    token_start = 0

    for token in tokens:
        token_end = token_start + len(token)
        token_labels = char_labels[token_start:token_end]

        # Determine the label for the token (use the first non-O label in the token's span)
        token_label = 'O'
        for label in token_labels:
            if label != 'O':
                token_label = label
                break

        final_tokens.append(token)
        final_labels.append(token_label)

        token_start = token_end
        # Move past any whitespace
        while token_start < len(sentence) and sentence[token_start].isspace():
            token_start += 1

    return final_tokens, final_labels


def _insert_label_gernermed(input_dict: dict, char_labels: List[str]) -> List[str]:
    for anno in input_dict['annotations']:
        char_labels[anno['de_spans'][0]] = f'B-{anno["type"]}'
        for i in range(anno['de_spans'][0] + 1, anno['de_spans'][1]):
            char_labels[i] = f'I-{anno["type"]}'
    return char_labels


def _insert_label_gptnermed(input_dict: dict, char_labels: List[str]) -> List[str]:
    # Assign NER classes to the corresponding character positions with IOB tagging
    for ner_class, start, stop in zip(input_dict['ner_labels']['ner_class'], input_dict['ner_labels']['start'], input_dict['ner_labels']['stop']):
        char_labels[start] = f'B-{ner_class}'
        for i in range(start + 1, stop):
            char_labels[i] = f'I-{ner_class}'
    return char_labels


def _insert_label_n2c2(input_dict: dict, char_labels: List[str]) -> List[str]:
    # Assign NER classes to the corresponding character positions with IOB tagging
    for anno in input_dict['annotations']:
        char_labels[anno['term_start']] = f'B-{anno["label"]}'
        for i in range(anno['term_start'] + 1, anno['term_end']):
            char_labels[i] = f'I-{anno["label"]}'
    return char_labels


def find_token_ent(token: str, pred_data: dict, entity_tpes: List[str]) -> str:
    for ent_type in entity_tpes:
        if pred_data['parsed_output'].get(ent_type):
            _ent = _assign_entity(token.lower(), pred_data['parsed_output'][ent_type], ent_type)
            if _ent:
                return _ent
    return "O"


def build_concept_dict(concepts: list) -> dict:
    """
    Transform structure of concept list to dict with following hierarchy
    {
        row_index: {
            token_index: {
                token: "Token-A",
                label: "B-Drug"
            } ...
        } ...
    }
    """
    def _add_to_dict(
            concept_dict: dict,
            row_indx: int,
            token: str,
            token_indx: int,
            label: str,
            prefix: str = 'B-'):
        ent_dict = {
            'token': token,
            'label': prefix + label
        }
        if row_indx in concept_dict:
            concept_dict[row_indx][token_indx] = ent_dict
        else:
            concept_dict[row_indx] = {
                token_indx: ent_dict
            }
    
    concept_dict = {}
    # split subtokens and add IOB schema
    for cncpt in [c for c in concepts if c['type'] in ['problem', 'treatment']]:
        row_indx = cncpt['row_index']
        if cncpt['first_token_index'] == cncpt['last_token_index']:
            cncpt['token_index'] = cncpt['first_token_index']
            _add_to_dict(
                concept_dict=concept_dict,
                row_indx=row_indx,
                token=cncpt['content'],
                token_indx=cncpt['first_token_index'],
                label=cncpt['type']
            )
        else:
            token_index = range(cncpt['first_token_index'], cncpt['last_token_index'] + 1)
            # check
            if len(cncpt['content'].split()) != len(token_index):
                raise ValueError(f'Something went wrong for concept: {cncpt}')
            tok_prefix = 'B-'
            for indx, token in zip(token_index, cncpt['content'].split()):
                _add_to_dict(
                    concept_dict=concept_dict,
                    row_indx=row_indx,
                    token=token,
                    token_indx=indx,
                    label=cncpt['type'],
                    prefix=tok_prefix
                )
                tok_prefix = 'I-'
    
    return concept_dict


def add_n2c2_annotations(test_data: list, anno_data: list):
    """
    Method adds annotations list of concepts to passed dataset.
    """

    def _has_same_annotations(sampel_ids: list, n2c2_data: list) -> bool:
        """Method to compare annotations with the same input text"""
        annos_ref = n2c2_data[sampel_ids.pop(0)]['annotations']
        for _id in sampel_ids:
            for i, anno in enumerate(n2c2_data[_id]['annotations']):
                if (
                    anno['label'] != annos_ref[i]['label']) or (
                        anno['term_start'] != annos_ref[i]['term_start']) or (
                            anno['content'] != annos_ref[i]['content']):
                    return False
        return True

    for test_record in test_data:
        _found_ids = []
        for anno_i, anno_record in enumerate(anno_data):
            if test_record['input'] == anno_record['input']:
                _found_ids.append(anno_i)
        if len(_found_ids) > 1:
            #if _has_relevant_entities(n2c2_2018_anno[_found_ids[0]]):
            #    raise ValueError('Found multiple ids for annotated text.', test_i)
            if not _has_same_annotations(_found_ids.copy(), anno_data):
                raise ValueError(f'Found multiple but different annotations in {_found_ids}')

        test_record['annotations'] = anno_data[_found_ids[0]]['annotations']
