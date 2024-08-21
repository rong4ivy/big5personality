import time
import os
import torch
import numpy as np
import json
import optuna
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, set_seed, get_scheduler, EvalPrediction, AdamW
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from datasets import Dataset, DatasetDict
from custom_model import CustomModel  # Import the custom model
from torch.nn import MSELoss
from nlpaug.augmenter.word import SynonymAug

start_time = time.time()

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
seed = 218
set_seed(seed)

data_path = "data/personality_train.json"  # Update this to the path of your dataset
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Function to load data
def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

# Load and format the data
data = load_data(data_path)

# Data augmentation
def augment_texts(texts):
    aug = SynonymAug(aug_src='wordnet')
    augmented_texts = [aug.augment(text) for text in texts]
    return augmented_texts

# Convert to a Hugging Face dataset object
def process_data_to_dataset(data):
    texts = [str(item['instruction']) for item in data]  # Ensure texts are strings
    outputs = [item['output'] for item in data]

    # Augment texts
    augmented_texts = augment_texts(texts)

    # Combine original and augmented texts
    combined_texts = texts + augmented_texts
    combined_outputs = outputs + outputs  # Duplicate outputs for the augmented texts

    # Parse each output into separate scores
    agreeableness = [float(output.split(';')[0].split(':')[1].strip()) for output in combined_outputs]
    openness = [float(output.split(';')[1].split(':')[1].strip()) for output in combined_outputs]
    conscientiousness = [float(output.split(';')[2].split(':')[1].strip()) for output in combined_outputs]
    extraversion = [float(output.split(';')[3].split(':')[1].strip()) for output in combined_outputs]
    neuroticism = [float(output.split(';')[4].split(':')[1].strip()) for output in combined_outputs]

    # Ensure all lists are of the same length
    assert len(combined_texts) == len(agreeableness) == len(openness) == len(conscientiousness) == len(extraversion) == len(neuroticism)

    # Convert to Python lists to ensure proper formatting
    agreeableness = list(map(float, agreeableness))
    openness = list(map(float, openness))
    conscientiousness = list(map(float, conscientiousness))
    extraversion = list(map(float, extraversion))
    neuroticism = list(map(float, neuroticism))

    # Debugging information
    #print(f"Combined texts (type: {type(combined_texts[0])}): {combined_texts[:2]}")  # Print first 2 for brevity
    #print(f"Agreeableness (type: {type(agreeableness[0])}): {agreeableness[:2]}")
    #print(f"Openness (type: {type(openness[0])}): {openness[:2]}")
    #print(f"Conscientiousness (type: {type(conscientiousness[0])}): {conscientiousness[:2]}")
    #print(f"Extraversion (type: {type(extraversion[0])}): {extraversion[:2]}")
    #print(f"Neuroticism (type: {type(neuroticism[0])}): {neuroticism[:2]}")

    dataset_dict = {
        'text': combined_texts,
        'agreeableness': list(map(float, agreeableness)),
        'openness': list(map(float, openness)),
        'conscientiousness': list(map(float, conscientiousness)),
        'extraversion': list(map(float, extraversion)),
        'neuroticism': list(map(float, neuroticism))
    }

    # Debugging to check the content of dataset_dict
    for key, value in dataset_dict.items():
        print(f"{key}: {type(value)}, length: {len(value)}, first element type: {type(value[0])}")

    # Ensure 'text' column is a list of strings
    dataset_dict['text'] = list(map(str, dataset_dict['text']))

    return Dataset.from_dict(dataset_dict)

# Process the data into a dataset
full_dataset = process_data_to_dataset(data)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_and_encode(examples):
    # Tokenizes the batch of texts
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    # Assemble the labels for the batch; each examples['*trait*'] is already a list of scores corresponding to the batch
    labels = torch.tensor([
        [agree, openn, consc, extrav, neuro]
        for agree, openn, consc, extrav, neuro in zip(
            examples['agreeableness'],
            examples['openness'],
            examples['conscientiousness'],
            examples['extraversion'],
            examples['neuroticism']
        )
    ]).float()  # Convert labels to float tensors

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Split the dataset into training, validation, and testing sets if not already done
train_test_split = full_dataset.train_test_split(test_size=0.1, seed=seed)  # 10% for testing
train_val_split = train_test_split['train'].train_test_split(test_size=0.1, seed=seed)  # 10% of the remaining 90% for validation

# Create a DatasetDict to hold the splits
dataset_dict = DatasetDict({
    'train': train_val_split['train'],
    'validation': train_val_split['test'],
    'test': train_test_split['test']
})

# Tokenize and encode datasets
train_dataset = dataset_dict['train'].map(tokenize_and_encode, batched=True)
val_dataset = dataset_dict['validation'].map(tokenize_and_encode, batched=True)
test_dataset = dataset_dict['test'].map(tokenize_and_encode, batched=True)

# Load the custom model
model = CustomModel(MODEL_NAME, num_labels=5).to(device)

# Custom Trainer class to define the loss function
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs[0]

        # compute custom loss (MSE loss)
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1, self.model.num_labels))

        return (loss, outputs) if return_outputs else loss

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score

def compute_metrics(p: EvalPrediction):
    labels = p.label_ids
    preds = p.predictions

    # Ensure the lengths are consistent
    if labels.shape[0] != preds.shape[0]:
        min_len = min(labels.shape[0], preds.shape[0])
        labels = labels[:min_len]
        preds = preds[:min_len]

    # Calculate regression metrics
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)

    # Convert predictions to binary format for classification metrics
    threshold = 0.5
    pred_classes = (preds >= threshold).astype(int)
    ref_classes = (labels >= threshold).astype(int)

    # Calculate classification metrics
    accuracy = accuracy_score(ref_classes.flatten(), pred_classes.flatten())
    f1 = f1_score(ref_classes.flatten(), pred_classes.flatten(), average='weighted')

    return {
        'mse': mse,
        'mae': mae,
        'accuracy': accuracy,
        'f1': f1
    }


def objective(trial):
    # Hyperparameters to be tuned
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 32, step=8)
    weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1, log=True)

    training_args = TrainingArguments(
        output_dir='./results2',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=8,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='mae',
        fp16=True,  # Enable mixed precision training for faster training on GPUs
        weight_decay=weight_decay,
        warmup_steps=100,
        logging_dir='./logs',
        logging_steps=50,
        learning_rate=learning_rate,
    )

    # Create optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        name='cosine_with_restarts',  # Use cosine scheduler with warm restarts
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataset) * 20
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),  # Specify custom optimizer and scheduler
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # Add early stopping with more patience
    )

    trainer.train()
    
    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    return eval_results['eval_mae']

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print(f"Best hyperparameters: {study.best_params}")

# Train final model with best hyperparameters
best_params = study.best_params

training_args = TrainingArguments(
    output_dir='./results2',
    per_device_train_batch_size=best_params['batch_size'],
    per_device_eval_batch_size=best_params['batch_size'],
    num_train_epochs=22,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='mae',
    fp16=True,
    weight_decay=best_params['weight_decay'],
    warmup_steps=100,
    logging_dir='./logs',
    logging_steps=50,
    learning_rate=best_params['learning_rate'],
)

optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'])
lr_scheduler = get_scheduler(
    name='cosine_with_restarts',
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_dataset) * 20
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()

trainer.save_model("./results2/personality_model4")  # save best model
tokenizer.save_pretrained("./results2/personality_model4")

end_time = time.time()

# Calculate the duration
duration = end_time - start_time

# Print the duration in a readable format
print(f"Training completed in {duration // 3600} hours, {(duration % 3600) // 60} minutes, and {duration % 60:.2f} seconds.")

