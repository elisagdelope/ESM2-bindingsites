import os
import wandb
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer
from datasets import Dataset
from accelerate import Accelerator
from peft import PeftModel

# Helper functions and data preparation
def truncate_labels(labels, max_length):
    """Truncate labels to the specified max_length."""
    return [label[:max_length] for label in labels]

def compute_metrics(p):
    """Compute metrics for evaluation."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove padding (-100 labels)
    predictions = predictions[labels != -100].flatten()
    labels = labels[labels != -100].flatten()
    
    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Compute precision, recall, F1 score, and AUC
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc = roc_auc_score(labels, predictions)
    
    # Compute MCC
    mcc = matthews_corrcoef(labels, predictions)
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mcc': mcc}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom compute_loss function."""
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss()
        active_loss = inputs["attention_mask"].view(-1) == 1
        active_logits = outputs.logits.view(-1, model.config.num_labels)
        active_labels = torch.where(
            active_loss, inputs["labels"].view(-1), torch.tensor(loss_fct.ignore_index).type_as(inputs["labels"])
        )
        loss = loss_fct(active_logits, active_labels)
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    # Environment setup
    accelerator = Accelerator()
    wandb.init(project='binding_site_prediction')
    
    # Load data and labels
    with open("600K_data/train_sequences_chunked_by_family.pkl", "rb") as f:
        train_sequences = pickle.load(f)
    with open("600K_data/test_sequences_chunked_by_family.pkl", "rb") as f:
        test_sequences = pickle.load(f)
    with open("600K_data/train_labels_chunked_by_family.pkl", "rb") as f:
        train_labels = pickle.load(f)
    with open("600K_data/test_labels_chunked_by_family.pkl", "rb") as f:
        test_labels = pickle.load(f)

    # Tokenization and dataset creation
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    max_sequence_length = tokenizer.model_max_length
    train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
    test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
    train_labels = truncate_labels(train_labels, max_sequence_length)
    test_labels = truncate_labels(test_labels, max_sequence_length)
    train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
    test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)

    # Load the pre-trained LoRA model
    base_model_path = "facebook/esm2_t12_35M_UR50D"
    lora_model_path = "esm2_t12_35M_lora_binding_sites_2023-09-21_17-50-58/checkpoint-84029" # Replace with the correct path to your LoRA model
    base_model = AutoModelForTokenClassification.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model = accelerator.prepare(model)

    # Define a function to compute metrics and get the train/test metrics
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, data_collator=data_collator, compute_metrics=compute_metrics)
    train_metrics = trainer.evaluate(train_dataset)
    test_metrics = trainer.evaluate(test_dataset)
    
    # Print the metrics
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics: {test_metrics}")

    # Log metrics to W&B
    wandb.log({"Train metrics": train_metrics, "Test metrics": test_metrics})
