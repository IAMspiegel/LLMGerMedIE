from os import listdir, path
from typing import Dict, List, Tuple, Optional

from nervaluate import Evaluator
from datasets import Dataset, load_dataset

from medicaNLP.bronco.fewshots import MISTRAL_FEWSHOT_PROMPT, LLAMA2_FEWSHOT_PROMPT, LEO_FEWSHOT_PROMPT


def bronco_mistral_fewshot_prompt(sample: dict) -> str:
    return f"""{MISTRAL_FEWSHOT_PROMPT}\n[INST] {sample['input']} [/INST]"""


def bronco_llama2_fewshot_prompt(sample: dict) -> str:
    return f"""{LLAMA2_FEWSHOT_PROMPT}\n[INST] {sample['input']} [/INST]"""


def bronco_leo_fewshot_prompt(sample: dict) -> str:
    return f"""{LEO_FEWSHOT_PROMPT} <|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n"""


def bronco_mistral_instruct_prompt(sample: dict) -> str:
    return f"""[INST]Extrahiere die Medikamente, Behandlung und Diagnosen aus dem folgendem Text. Gib die gefundenen Entitäten als Liste aus.
{sample['input']} [/INST]"""


def bronco_instruction_prompt(sample: dict) -> str:
    return f"""### Instruction:
Extrahiere die Medikamente, Behandlung und Diagnosen aus dem folgendem Text. Gib die gefundenen Entitäten als Liste aus.

### Input:
{sample['input']}

### Response:
"""


def bronco_instruction_prompt2(sample: dict) -> str:
    return f"""### Instruction:
Extrahiere die Medikamente, Behandlung und Diagnosen aus dem folgendem Text, eine Diagnose ist eine Krankheit, Symptom oder eine medizinische Beobachtung (ICD-10). Eine Behandlung ist ein diagnostisches Verfahren, Operation oder systemische Krebsbehandlung (OPS). Gib die gefundenen Entitäten als Liste aus.

### Input:
{sample['input']}

### Response:
"""


def parse_output(output: str, entities: List[str], early_break: bool = False) -> dict:
    """Method to parse instruction finetuned output"""
    entities_dict: dict = {}

    for line in output.split('\n'):
        for ent in entities:
            if line.strip().startswith(ent):
                # quick workaround
                if ent == 'Diagnose' and line.startswith('Diagnosen'):
                    ent_output = line.replace('Diagnosen:', '').replace('\n', '')
                else:
                    ent_output = line.replace(f'{ent}:', '').replace('\n', '')
                if ent_output.strip():
                    entities_dict[ent] = [e.strip() for e in ent_output.split(',')]
                else:
                    entities_dict[ent] = None
        # check if all entities were found
        if early_break:
            if all([e in entities_dict.keys() for e in entities]):
                break
    return entities_dict


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
    for e_key, e_type in [('Medikamente', 'MED'), ('Diagnose', 'DIAG'), ('Behandlung', 'TREAT')]:

        if pred_data_dict['parsed_output'].get(e_key):
            _ent = _assign_entity(token.lower(), pred_data_dict['parsed_output'][e_key], e_type)
            if _ent:
                return _ent
    return "O"


def build_mistral_fewshot(
    instruction: str,
    fewshots: List[dict]
) -> str:
    first_example = True
    pre_prompt = f"<s>[INST] {instruction}\n"
    for i, f_shot in enumerate(fewshots):
        if first_example:
            pre_prompt = pre_prompt + f"{f_shot['input']} [\INST]\n{f_shot['response']}\n"
            first_example = False
        else:
            if i < len(fewshots) - 1:
                pre_prompt = pre_prompt + f"[INST] {f_shot['input']} [\INST]\n{f_shot['response']}\n"
            # add first sentence end token
            else:
                pre_prompt = pre_prompt + f"[INST] {f_shot['input']} [\INST]\n{f_shot['response']}</s>\n"
    return pre_prompt


def build_leo_fewshot(
    system_message: Optional[str],
    instruction: str,
    fewshots: List[dict],
    user_assistan_in_example: bool = False
) -> str:
    first_example = True
    pre_prompt = ""
    # add system message
    if system_message:
        pre_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    # add instruction
    pre_prompt = pre_prompt + f"<|im_start|>user\n{instruction}\n"
    # add few shots
    for shot in fewshots:
        if first_example:
            pre_prompt = pre_prompt + f"{shot['input']}<|im_end|>\n<|im_start|>assistant\n{shot['response']}<|im_end|>\n"
            first_example = False
        else:
            pre_prompt = pre_prompt + f"<|im_start|>user{shot['input']}<|im_end|>\n<|im_start|>assistant\n{shot['response']}<|im_end|>\n"
    return pre_prompt 

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


def get_raw_texts(file_dir: str, key_sep: str = "-") -> Dict:
    """
    This method collects all sentences from files in the given directory.
    The sentences are stored in a dictionary, where key is the `file_name` + `sentence_index`.
    :param file_dir: path of directory containing text files
    :param key_sep: sparator of filename and sentence index
    """
    file_sentences: dict = {}

    for file in listdir(file_dir):
        with open(path.join(file_dir, file), "r") as fh:
            for indx, line in enumerate(fh.readlines()):
                # append sentence without '\n'
                file_sentences[file + key_sep + str(indx + 1)] = line[:-1]

    return file_sentences


def build_input_response_hugginface_dataset(
    hf_dataset: Dataset,
    data_split: str,
    sentence_dict: dict,
    entities: List[Tuple[str, str]]
) -> Dataset:
    """Wrapper method for `build_input_response_dataset` to build a hugginface"""

    # build dataset
    hf_data = build_input_response_dataset(
        hf_dataset=hf_dataset,
        data_split=data_split,
        sentence_dict=sentence_dict,
        entities=entities
    )

    def dataset_generator():
        for data_dict in hf_data:
            yield data_dict

    # generate huggingface.Dataset
    hf_data = Dataset.from_generator(dataset_generator)

    return hf_data


def build_input_response_dataset(
    hf_dataset: Dataset,
    data_split: str,
    sentence_dict: dict,
    entities: List[Tuple[str, str]]
) -> List[dict]:
    """
    This method builds a dataset containing of lists and dictionaries
    """

    instruction_dataset: List[dict] = []

    for data_record in hf_dataset[data_split]:
        # get input sentence
        _key = data_record['fname'].replace('CONLL', 'txt') + '-' + str(data_record['sentence_id'])
        sentence = sentence_dict[_key]

        # build response
        record_ents = {e: [] for e, _ in entities}
        ent_tags = [e for e, _ in entities]

        def __get_complete_word(tag: str, token_indx: int, tokens: List) -> List:
            """
            Helper method to combine IOB scheme tokens to one list
            """
            if len(data_record['tags']) >= token_indx + 2:
                try:
                    if data_record['tags'][token_indx + 1] == 'I-' + tag:
                        tokens.append(data_record['tokens'][token_indx + 1])
                        __get_complete_word(tag, token_indx + 1, tokens)
                except IndexError as indxerr:
                    print(len(data_record['tags']))
                    print(token_indx)
                    print('****')
                    print(data_record)
                    raise indxerr
            return tokens

        for i, t in enumerate(data_record['tags']):
            if t.startswith('B-') and t.replace('B-', '') in ent_tags:
                ent = t.replace('B-', '')
                entity_tokens = __get_complete_word(
                    tag=ent,
                    token_indx=i,
                    tokens=[data_record['tokens'][i]]
                )
                # append to entity list
                if record_ents[ent]:
                    entity_list = record_ents[ent].copy()
                    entity_list.append(' '.join(entity_tokens))
                else:
                    entity_list = [' '.join(entity_tokens)]
                record_ents[ent] = entity_list
        
        # build response
        response = ""
        for ent, ent_label in entities:
            if record_ents[ent]:
                response += f"{ent_label}: {','.join(record_ents[ent])}\n"
            else:
                response += f"{ent_label}:\n"

        # append to list
        instruction_dataset.append({
            "id": _key,
            "input": sentence,
            "response": response,
            "tags": data_record['tags'],
            "tokens": data_record['tokens']
        })

    return instruction_dataset
