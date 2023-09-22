import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
from accelerate import Accelerator
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import pickle

# Initialize accelerator and Weights & Biases
accelerator = Accelerator()
os.environ["WANDB_NOTEBOOK_NAME"] = 'train.py'
wandb.init(project='binding_site_prediction')

# Helper Functions and Data Preparation
def save_config_to_txt(config, filename):
    """Save the configuration dictionary to a text file."""
    with open(filename, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def truncate_labels(labels, max_length):
    return [label[:max_length] for label in labels]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions[labels != -100].flatten()
    labels = labels[labels != -100].flatten()
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc = roc_auc_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mcc': mcc}

def compute_loss(model, inputs):
    logits = model(**inputs).logits
    labels = inputs["labels"]
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    active_loss = inputs["attention_mask"].view(-1) == 1
    active_logits = logits.view(-1, model.config.num_labels)
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    loss = loss_fct(active_logits, active_labels)
    return loss

# Load data from pickle files
with open("600K_data/train_sequences_chunked_by_family.pkl", "rb") as f:
    train_sequences = pickle.load(f)
    
with open("600K_data/test_sequences_chunked_by_family.pkl", "rb") as f:
    test_sequences = pickle.load(f)

with open("600K_data/train_labels_chunked_by_family.pkl", "rb") as f:
    train_labels = pickle.load(f)

with open("600K_data/test_labels_chunked_by_family.pkl", "rb") as f:
    test_labels = pickle.load(f)
    
# Tokenization
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

# Set max_sequence_length to the tokenizer's max input length
max_sequence_length = tokenizer.model_max_length

train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)

# Directly truncate the entire list of labels
train_labels = truncate_labels(train_labels, max_sequence_length)
test_labels = truncate_labels(test_labels, max_sequence_length)

train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)

# Compute Class Weights
classes = [0, 1]  
flat_train_labels = [label for sublist in train_labels for label in sublist]
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(accelerator.device)

# Define Custom Trainer Class
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = compute_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss

# Define and run training function
def train_function_no_sweeps(train_dataset, test_dataset):
    
    # Directly set the config
    config = {
        "lora_alpha": 1, 
        "lora_dropout": 0.4,
        "lr": 5.701568055793089e-04,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 6,
        "r": 1,
        "weight_decay": 0.4,
        # Add other hyperparameters as needed
    }

    # Log the config to W&B
    wandb.config.update(config)

    # Save the config to a text file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config_filename = f"esm2_t12_35M_lora_config_{timestamp}.txt"
    save_config_to_txt(config, config_filename)
    
    model_checkpoint = "facebook/esm2_t12_35M_UR50D"  
    
    # Define labels and model
    id2label = {0: "No binding site", 1: "Binding site"}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)

    # Convert the model into a PeftModel
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, 
        inference_mode=False, 
        r=config["r"], 
        lora_alpha=config["lora_alpha"], 
        target_modules=["query", "key", "value"], # also maybe "dense_h_to_4h" and "dense_4h_to_h"
        lora_dropout=config["lora_dropout"], 
        bias="none" # or "all" or "lora_only" 
    )
    model = get_peft_model(model, peft_config)

    # Use the accelerator
    model = accelerator.prepare(model)
    train_dataset = accelerator.prepare(train_dataset)
    test_dataset = accelerator.prepare(test_dataset)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Training setup
    training_args = TrainingArguments(
        output_dir=f"esm2_t12_35M_lora_binding_sites_{timestamp}",
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler_type"],
        gradient_accumulation_steps=1,
        max_grad_norm=config["max_grad_norm"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=None,
        logging_first_step=False,
        logging_steps=200,
        save_total_limit=7,
        no_cuda=False,
        seed=8893,
        fp16=True,
        report_to='wandb'
    )

    # Initialize Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # Train and Save Model
    trainer.train()
    save_path = os.path.join("lora_binding_sites", f"best_model_esm2_t12_35M_lora_{timestamp}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

# Call the training function
if __name__ == "__main__":
    train_function_no_sweeps(train_dataset, test_dataset)
