from typing import Tuple, Union, List
import os
import time
import json

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset

from medicaNLP.bronco.bronco_utils import bronco_instruction_prompt, parse_output, bronco_mistral_instruct_prompt, bronco_instruction_prompt2
from medicaNLP.cardiode.build_hf_dataset import cardio_instruction_prompt
from medicaNLP.ggponc.hf_utils import ggponc_instruction_prompt

SUPPORTED_DATASETS = ['cardiode', 'bronco', 'ggponc']

# *** NOTE ****
# make sure that you change all file paths, entities
# *** *** ***


def predict_data(
    model_path: str,
    data_path: str,
    dataset_name: str,
    extract_entities: List[str],
    pred_file_path: str,
    top_p: float = 0.9,
    temperature: float = 0.1,
    max_new_tokens: int = 200,
    use_flash: bool = False
):
    """Wrapper method to generate output on test dataset with LLM"""
    # check dataset name
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f'Unknown dataset_name {dataset_name}, supported are {SUPPORTED_DATASETS}.')

    # init model and tokenizer
    llm, tokenizer = get_model_and_tokenizer(model_path, use_flash)
    # load dataset
    test_dataset = load_dataset("json", data_files={"test": data_path})
    start = time.time()
    # generate response for records in dataset
    pred_test_data = get_parsed_output(
        dataset_name,
        test_dataset['test'],
        tokenizer,
        llm,
        extract_entities,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    end = time.time()

    pred_datadict = {
        "model": model_id,
        "generation_cfg": {
            "load_in_4b": True,
            "top_p": top_p,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        },
        "prediction": pred_test_data,
        "time": str(end - start)
    }

    with open(pred_file_path, "w") as fp:
        json.dump(pred_datadict, fp)

    print(f'Prediction took {end - start} sec')


def get_parsed_output(
    dataset_name: str,
    hf_data: Dataset,
    tokenizer: AutoTokenizer,
    llm: Union[AutoModelForCausalLM, AutoPeftModelForCausalLM],
    extract_entities: List[str],
    top_p: float = 0.9,
    temperature: float = 0.1,
    max_new_tokens: int = 200
) -> List[dict]:
    """Method to generate output for given dataset with LLM"""
    # var to store parsed output
    parsed_predictions: List[dict] = []

    count = 1
    data_len = len(hf_data)

    for record in hf_data:
        
        print(f"{count}/{data_len}")
        count += 1
        
        # create prompt
        if dataset_name.lower() == 'bronco':
            prompt = bronco_instruction_prompt(record)
            r_id = record['id']
        elif dataset_name.lower() == 'cardiode':
            prompt = cardio_instruction_prompt(record, input_key='text')
            r_id = record['fname'] + str(record['sentence_id'])
        elif dataset_name.lower() == 'ggponc':
            prompt = ggponc_instruction_prompt(record)
            r_id = record['fname'] + str(record['sentence_id'])

        # tokenize, predict and decode
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        outputs = llm.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature
        )
        output_str = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
        

        # append data
        output_dict = {
            "id": r_id,
            "raw_output": output_str
        }

        try:
            output_dict['parsed_output'] = parse_output(output_str, extract_entities)
        except KeyboardInterrupt as keyint:
            raise keyint
        except SystemExit as sysexit:
            raise sysexit
        except:
            output_dict['parsed_output'] = ""

        parsed_predictions.append(output_dict)

    return parsed_predictions


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


if __name__ == "__main__":
    # *** NOTE ****
    # make sure that you change all file paths, entities
    # *** *** ***

    model_id = '/path/to/peft_mistral_7B'
    model_name = os.path.basename(model_id)

    predict_data(
        model_path=model_id,
        data_path='/path/to/instr_test.json',
        dataset_name='bronco',
        pred_file_path=f'/path/to/pred_{model_name}_instr_test.json',
        extract_entities=['Medikamente', 'Diagnose', 'Behandlung'],
        top_p=0.95,
        temperature=0.1,
        max_new_tokens=240,
        use_flash=False
    )
