from typing import List, Tuple, Dict
import json
import re
import os
from datasets import Dataset, load_dataset


GERNERMED_INSTRUCTION = "Extrahiere alle Medikamente aus dem folgendem Text. Gib die gefundenen Entitäten als Liste aus."
GPTNERMED_INSTRUCTION = "Extrahiere alle Medikamente und Diagnosen aus dem folgendem Text. Gib die gefundenen Entitäten als Liste aus."
I2B2_2010_INSTRUCTION = "Extract all possbile medical problems and treatments from the following text. List them in a comma sperated list."
N2C2_2018_INSTRUCTION = "Extract all drugs from the follwoing text. If the text mentions reasons or adverse drug events (ADE) for the mentioned drugs extract them, too. List the entities in a comma sperated list."


def dataset_to_huggingface_dataset(data: List[dict]) -> Dataset:
    """Helper method to build Hugginface Dataset from list of dictionaries"""

    def dataset_generator():
        for data_dict in data:
            yield data_dict

    return Dataset.from_generator(dataset_generator)


def build_instruction_dataset(file_dict: dict) -> List[dict]:

    supported_datasets = ['gernermed', 'i2b2_2010']

    instruction_dataset: List[dict] = []

    for dataset_name, file_path in file_dict.items():
        if dataset_name.lower() in supported_datasets:
            if dataset_name.lower() == "gernermed":
                instruction_dataset += _build_gernermed_instruct(file_path)
            elif dataset_name.lower() == 'i2b2_2010':
                instruction_dataset += build_i2b2_2010_instruct_dataset(file_path)
            else:
                print(f'{dataset_name} unknwon dataset, skip file.')

    return instruction_dataset


def _build_gernermed_instruct(file_path: str) -> List[dict]:
    """
    wrapper method to iterate over entry in dataset and build instruction, input, response dataset structure
    """
    _instruct_data: List[dict] = []
    # read json data
    with open(file_path, 'r') as fh:
        gernermed_data = json.load(fh)

    for entry in gernermed_data:
        resp_txt, drug_tokens = _build_gernermed_response_list(entry['de'], entry['annotations'])
        _instruct_data.append(
            {
                "instruction": GERNERMED_INSTRUCTION,
                "input": entry['de'],
                "response": resp_txt,
                "tokens": drug_tokens
            }
        )

    return _instruct_data


def _build_gernermed_response_list(de_txt: str, annotations: List[dict]) -> Tuple[str, List[str]]:
    # collect drug entities
    response_txt = "Medikamente: "
    drugs: List[str] = []
    # iterate over annotations, which are sorted by start span in the german text
    for ann in sorted(annotations, key=lambda d: d['de_spans'][0]):
        if ann['type'] == 'Drug':
            token = de_txt[ann['de_spans'][0]:ann['de_spans'][1]]
            # clean token from punctuation, note we do not want to remove `-`
            cleaned_token = token.translate(str.maketrans('', '', '!.,?:;@#$&*'))
            # append to list
            drugs.append(cleaned_token)
    # build string
    if drugs:
        response_txt = response_txt + ", ".join(drugs)
    return response_txt, drugs


def build_i2b2_2010_instruct_dataset_from_file(
    file_name: str,
    ann_dir: str,
    text_dir: str,
    instruction: str = I2B2_2010_INSTRUCTION,
    max_lines: int = 8
) -> List[dict]:
    """
    Wrapper method to process text and annotation file to instruction dataset
    """
    # read text file
    text_lines = []
    with open(os.path.join(text_dir, file_name + '.txt'), "r") as fh:
        for line in fh.readlines():
            text_lines.append(line)

    # read annotations
    concepts = []
    with open(os.path.join(ann_dir, file_name + '.con'), "r") as fh:
        for line in fh.readlines():
            # parse concept string to dict
            concepts.append(_parse_i2b22010_concept_annotation(line))

    # match concepts to text lines
    text_with_concepts = match_i2b22010_concepts_with_text(text_lines, concepts)

    # split into smaller sections
    sections_with_concepts = split_i2b22010_into_sections(text_with_concepts, max_lines_per_section=max_lines)

    # create instructions, response dataset from sections
    instruct_dataset = build_i2b22010_instruction_dataset_for_sections(file_name, instruction, sections_with_concepts)

    return instruct_dataset


def _parse_i2b22010_concept_annotation(row: str) -> dict:
    # extract the concept
    match = re.search(r'c="([^"]*)"', row)
    if not match:
        raise RuntimeError(f"Cannot extract concept string from row: {row}")

    entity_content = match.group(1)

    # catch double quotes within the concept string -> failed to extract concept correctly
    if not entity_content:
        # get last double quote of concept string
        quote_index = row.split('||t="')[0].rfind('"')
        # remove double quotes and clean string
        entity_content = row[:quote_index].replace('c=', '').replace('"', '').strip().replace('  ', ' ')
        concept_end = quote_index + 1
    else:
        concept_end = match.end()

    conc_loc, conc_type = row[concept_end:].split('||', maxsplit=1)

    # extract the location in text file from concept
    loc_1, loc_2 = conc_loc.strip().split(' ', maxsplit=1)
    if loc_1.split(':')[0] != loc_2.split(':')[0]:
        print(row)
        print(loc_1, loc_2)
        raise RuntimeError(f"Row index does not match for concept row. {row}")

    # extract the type from concept
    type_match = re.search(r't="([^"]*)"', conc_type)
    if not type_match:
        raise RuntimeError(f"Cannot extract concept type from row. {row}")
    entity_type = type_match.group(1)

    return {
        "content": entity_content,
        "type": entity_type,
        "row_index": int(loc_1.split(':')[0]),
        "first_token_index": int(loc_1.split(':')[1]),
        "last_token_index": int(loc_2.split(':')[1]),
    }


def match_i2b22010_concepts_with_text(text_lines: List[str], concepts: List[dict]) -> List[dict]:
    # match concepts with text lines
    text_with_concepts = [{"text": t, "index": i + 1, "concepts": []} for i, t in enumerate(text_lines)]
    for con in concepts:
        # append concept to list of concepts
        entry = text_with_concepts[con['row_index'] - 1]
        if not entry['concepts']:
            exisinting_concepts = [con]
        else:
            exisinting_concepts = entry['concepts'].copy()
            exisinting_concepts.append(con)
        # replace list of concepts
        text_with_concepts[con['row_index'] - 1]['concepts'] = exisinting_concepts

    # check if all concepts were matched to text lines
    added_concs = [c for e in text_with_concepts if e['concepts'] for c in e['concepts']]
    if len(added_concs) != len(concepts):
        raise RuntimeError(
            f"Length of given concepts {len(concepts)} does not match len of matched concepts{len(added_concs)}."
        )

    return text_with_concepts


# create chunks from text file, since the text length exceeds the 2048 length

def split_i2b22010_into_sections(text_with_concepts: List[dict], max_lines_per_section: int = 8, max_len: int = 2048):
    """
    Method to split text file into sections and match the concepts of that text to the section.
    Indicator for a new section is `:\n` or the count of text lines.
    """
    sections_with_concepts: List[dict] = []

    # helper vars
    temp_section: List[dict] = []
    _line_count = 0

    for i, txt_line in enumerate(text_with_concepts):

        # end condition
        if i == len(text_with_concepts) - 1:
            temp_section.append(txt_line)
            sections_with_concepts.append(temp_section)
        else:
            # start of new section
            if txt_line['text'].endswith(':\n') or _line_count == max_lines_per_section:
                # new section -> process section before
                if temp_section:
                    sections_with_concepts.append(temp_section)
                # create new temp section
                temp_section = [txt_line]
                _line_count = 0
            else:
                temp_section.append(txt_line)
                _line_count += 1

    return sections_with_concepts


def build_i2b22010_instruction_dataset_for_sections(
    doc_id: str,
    instruction: str,
    sections_with_concepts: List[dict],
    max_len: int = 2048
) -> List[dict]:
    """
    Method builds input and response string for each section
    """
    instruction_dataset: List[dict] = []

    for section in sections_with_concepts:

        section_txt = ""
        section_problems = []
        section_treatments = []
        section_concepts = []

        for idx, line in enumerate(section):
            # build string
            section_txt += line['text']
            # append concepts to lists
            if line['concepts']:
                for concept in line['concepts']:
                    if concept['type'] == 'problem':
                        section_problems.append(concept['content'])
                    elif concept['type'] == 'treatment':
                        section_treatments.append(concept['content'])
            section_concepts += line['concepts']
            # store start and end of section
            if idx == 0:
                section_start = line['index']
                # check if section contains only one line
                if len(section) == 1:
                    section_end = line['index']
            elif idx == len(section) - 1:
                section_end = line['index']

        # build response string
        section_response = f"""problem: {', '.join(section_problems)}\ntreatment: {', '.join(section_treatments)}"""

        # check length of string
        if len(section_txt) + len(section_response) + len(instruction) + len('###Response\n###Input') > max_len - 10:
            raise RuntimeError(
                f"Section from {doc_id} is to long, len: {len(section_txt)}. {section_start}:{section_end}"
            )

        # append to dataset
        instruction_dataset.append({
            "doc": doc_id,
            "start": section_start,
            "end": section_end,
            "input": section_txt,
            "response": section_response,
            "concepts": section_concepts
        })

    return instruction_dataset


def build_gptnermed_instruct_data(dataset: Dataset, use_labels: List[int] = [0, 2]) -> List[dict]:
    """Method to build input response dict for instruction fine tuning"""

    ner_label_dict = {
        0: 'Medikamente',
        1: 'Dosis',
        2: 'Diagnose'
    }

    instruc_data: List[dict] = []
    for i, record in enumerate(dataset):
        instruc_data.append({
            'id': i,
            'instruction': GPTNERMED_INSTRUCTION,
            'input': record['sentence'],
            'response': _build_gptnermed_response(record, use_labels, ner_label_dict)
        })

    return instruc_data


def _build_gptnermed_response(record: dict, use_labels: List[int], ner_label: Dict[int, str]) -> str:

    # helper var to store ner_labels and content
    entity_dict = {k: [] for k in use_labels}

    # collect ner labels and content
    for ner_cls, start, end in zip(
        record['ner_labels']['ner_class'],
        record['ner_labels']['start'],
        record['ner_labels']['stop']
    ):
        if ner_cls in use_labels:
            entity_dict[ner_cls].append(record['sentence'][start:end])

    # build response string
    response_str = ""
    for k, v in entity_dict.items():
        response_str = response_str + f"{ner_label[k]}: {', '.join(v)}\n"

    return response_str
