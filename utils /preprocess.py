import json
import requests
from datasets import Dataset
import torch
from transformers import AutoTokenizer
import config

def load_data(url):
    if url.startswith('http'):
        response = requests.get(url)
        data = response.json()
    else:
        with open(url, 'r') as file:
            data = json.load(file)
    return data


def tokenize_and_encode(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64)
    labels = torch.tensor([
        [agree, openn, consc, extrav, neuro]
        for agree, openn, consc, extrav, neuro in zip(
            examples['agreeableness'],
            examples['openness'],
            examples['conscientiousness'],
            examples['extraversion'],
            examples['neuroticism']
        )
    ]).float()

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def preprocess_data():
    data = load_data(config.DATA_URL)
    full_dataset = process_data_to_dataset(data)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL, use_fast=True)

    train_test_split = full_dataset.train_test_split(test_size=0.1)
    train_val_split = train_test_split['train'].train_test_split(test_size=0.1)

    dataset_dict = {
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': train_test_split['test']
    }

    train_dataset = dataset_dict['train'].map(lambda x: tokenize_and_encode(x, tokenizer), batched=True)
    val_dataset = dataset_dict['validation'].map(lambda x: tokenize_and_encode(x, tokenizer), batched=True)
    test_dataset = dataset_dict['test'].map(lambda x: tokenize_and_encode(x, tokenizer), batched=True)

    return train_dataset, val_dataset, test_dataset, tokenizer

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, tokenizer = preprocess_data()
    print("Preprocessing completed.")
