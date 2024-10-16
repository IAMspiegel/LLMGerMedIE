# LLMGerMedIE
Repository to reproduce clinical information extraction experiments in German with LLMs. This repository contains pipelines to instruction tune any LLM with the QLoRA method.

**DISCLAIMER:** Note, that this repository does not contain any fine-tuned transformer models or dataset specific few-shot prompts due to data protection restrictions.

## Repository Structure
The package is structured as follows:
- **bronco**: `BRONCO150` related dataset preprocessing and LLM postprocessing. Exemplary notebook using vLLM.
- **cardiode**: `CARDIO:DE` related dataset preprocessing
- **ggponc**: `GGPONC v2` related dataset preprocessing
- **instructionTuning**: scripts to train (`train_instruction.py`) and predict (`pred_data.py`) LLMs with predefined instruction datasets. Moreover, methods to transform NER datasets into instruction tuning ready format, see below.

## Datasets
In order to reproduce results and instruction tune LLMs the following datasets must be acquired:
#### Instruction Tuning Dataset
- <b>GERNERMED</b> https://github.com/frankkramer-lab/GERNERMED/tree/main/data
- <b>GPTNERMED</b> https://huggingface.co/datasets/jfrei/GPTNERMED
- <b>n2b2 2018</b> https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- <b>i2b2 2010</b> https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
#### Test Datasets
- <b>BRONCO</b> https://www2.informatik.hu-berlin.de/~leser/bronco/index.html
- <b>GGPONCv2</b> https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/
- <b>CARDIO:DE</b> https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/AFYQDY

## Transform Data
All methods to transform NER datasets into the instruction schema (**Instruction**, **Input**, **Response**) can be found under `medicaNLP.instructionTuning.build_instruct_data`.
The stored huggingface instruction datasets can be used to fine-tune any LLM.

```python
# Example to transform datasets

from medicaNLP.instructionTuning.build_instruct_data import build_gptnermed_instruct_data, dataset_to_huggingface_dataset

# load huggingface dataset
hf_dataset = load_dataset("jfrei/GPTNERMED")

# transform into instruction schema
train_instruct = build_gptnermed_instruct_data(hf_dataset['train'])

# transform to hugginface dataset
train_instruct = dataset_to_huggingface_dataset(train_instruct)
```

## Install
Activate python environment
```bash
source venv/bin/activate
```
Move to repository directory and install requirements
```bash
cd path/to/LLMGerMedIE
pip install -r requirements.txt
```

Install medicaNLP
```bash
pip install -e .
```

## Instruction Tuning
To fine tune the LLM use the `medicaNLP.instructionTuning.train_instruction.py` script. Make sure to configure the relevant model and data paths. The script can be run in the background as follows:
```bash
nohup python3 train_instruction.py &
```
To investigate training validation use the methods in `medicaNLP.instructionTuning.validate_train.py`.

## Predict
To extract clinical information from a given data sample with an LLM use
```bash
nohup python3 medicaNLP.instructionTuning.pred_data.py &
```
To evaluate the extracted entities based on test samples use the dataset specific methods in `medicaNLP.instructionTuning.evaluate.py`.
