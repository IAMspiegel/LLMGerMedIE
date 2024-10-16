from typing import List, Tuple
from os import path, listdir

from datasets import Dataset


def cardio_instruction_prompt(sample: dict, input_key: str = 'input') -> str:
    return f"""### Instruction:
Extrahiere die Medikamente und Arzneistoffe aus dem folgendem Text. Gib die gefundenen Entitäten als Liste aus.

### Input:
{sample[input_key]}

### Response:
"""


def build_dataset_from_dir(
        file_dir: str,
        sentence_col: int,
        token_col_indx: int,
        tag_col_indx: int,
        n_columns: int,
        add_iob_scheme: bool = False
) -> Dataset:
    """
    Wrapper method for `file_to_hf_json_format()` and generates a huggingface Dataset.
    """
    hf_data: List[dict] = []
    # run through files in directory and append data
    for file in listdir(file_dir):
        hf_data += file_to_hf_json_format(
            path.join(file_dir, file),
            sentence_col=sentence_col,
            token_col_indx=token_col_indx,
            tag_col_indx=tag_col_indx,
            n_columns=n_columns,
            build_iob_scheme=add_iob_scheme
        )

    def dataset_generator():
        for data_dict in hf_data:
            yield data_dict

    # generate huggingface.Dataset
    hf_dataset = Dataset.from_generator(dataset_generator)

    return hf_dataset


def file_to_hf_json_format(
        file_path: str,
        sentence_col: int,
        token_col_indx: int,
        tag_col_indx: int,
        n_columns: int,
        build_iob_scheme: bool = False
) -> List[dict]:
    """
    Method to extract tokens and tags from tsv file and build a huggingface conform dictionary from columns.
    :param file_path: path to file
    :param sentence_col: column index for the sentence-token id. 1-1 -> sentence 1 and token 1
    :param token_col_indx: column index for token string
    :param tag_col_indx: column index for tag sring. Empty column is `_`
    :param n_columns: number of columns for each tsv row
    :param build_iob_scheme: flag if annotations should be transformed in IOB format
      DRUG -> B-DRUG
      DRUG[1], DRUG[1] -> B-DRUG, I-DRUG
    :return list of dictionary {'fname': str, 'sentence_id': str, 'tokens': List[str], 'tags': List[str]}
    """

    def _get_tag(tag: str, keep_brackets: bool = False) -> str:
        """
        Method to return entity label from tag string
        """
        if tag is None or tag == "_":
            return "O"
        if keep_brackets:
            return tag
        return tag.split('[')[0]

    # var to collect dictionary per sentence
    sentence_dicts: List[dict] = []
    # get file name
    fname = path.basename(file_path)

    # helper vars to store currently processed data
    current_sentence: int = 0
    sentence_tokens: List[str] = []
    sentence_tags: List[str] = []
    sentence_texts: List[str] = []

    # run through file
    with open(file_path, "r") as fh:
        for line in fh:
            if line.startswith("#Text="):
                sentence_texts.append(line.replace('#Text=', '')[:-1])
            elif not line == "\n" and not line.startswith("#"):
                cols = line.split()
                if len(cols) != n_columns:
                    raise RuntimeError(
                        f"Unexpected number of columns. Found {len(cols)} cols but expects {n_columns} in {line}"
                    )

                # extract info from tsv row
                sentence_id = int(cols[sentence_col].split('-', maxsplit=2)[0])

                if sentence_id == current_sentence:
                    sentence_tags.append(_get_tag(cols[tag_col_indx], keep_brackets=build_iob_scheme))
                    sentence_tokens.append(cols[token_col_indx])

                if sentence_id > current_sentence:
                    if current_sentence > 0:

                        # store previous collected data
                        sentence_dicts.append({
                            "fname": fname,
                            "sentence_id": current_sentence,
                            "tokens": sentence_tokens,
                            "tags": _add_IOB(sentence_tags) if build_iob_scheme else sentence_tags,
                            "text": sentence_texts.pop(0)
                        })
                    # process new sentence
                    sentence_tokens = [cols[token_col_indx]]
                    sentence_tags = [_get_tag(cols[tag_col_indx], keep_brackets=build_iob_scheme)]

                    current_sentence = sentence_id
                    
        # store last entry
        sentence_dicts.append({
            "fname": fname,
            "sentence_id": current_sentence,
            "tokens": _add_IOB(sentence_tags) if build_iob_scheme else sentence_tags,
            "tags": sentence_tags,
            "text": sentence_texts.pop(0)
        })

    return sentence_dicts


def _add_IOB(tags: List[str]):
    """
    Method to add 'B' or 'I' tag to annotated tags from Cardio:DE
    e.g.
        # Input
        [DRUG[1], DRUG[1], FREQ[1], FREQ[2], O, DRUG]
        
        # Output
        [B-DRUG, I-DRUG, B-FREQ, I-FREQ, O, B-DRUG]
    """
    # helper vars
    tag_id: int = 0
    iob_tags: List[str] = []

    for t in tags:
        if '[' in t:
            t_label, t_id = t.split('[', maxsplit=1)
            t_id = int(t_id.replace(']', ''))
            # check if beginning
            if t_id > tag_id:
                iob_tags.append('B-' + t_label)
                tag_id = t_id
            elif t_id == tag_id:
                iob_tags.append('I-' + t_label)
            else:
                raise RuntimeError(f'Could not determin IOB tag for {t} in {tags}.')
        else:
            iob_tags.append('B-' + t if t != 'O' else t)

    return iob_tags


def build_input_response_dataset(
        hf_dataset: Dataset,
        entities: List[Tuple[str, str]]
) -> List[dict]:
    """
    """

    instruction_dataset: List[dict] = []

    for data_record in hf_dataset:

        # build response
        record_ents = {e: [] for e, _ in entities}
        ent_tags = [e for e, _ in entities]

        