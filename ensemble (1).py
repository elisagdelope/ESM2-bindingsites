import os
import pickle
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset, concatenate_datasets
from accelerate import Accelerator
from peft import PeftModel
import gc

# Step 1: Load train/test data and labels from pickle files
with open("/kaggle/input/550k-dataset/train_sequences_chunked_by_family.pkl", "rb") as f:
    train_sequences = pickle.load(f)
with open("/kaggle/input/550k-dataset/test_sequences_chunked_by_family.pkl", "rb") as f:
    test_sequences = pickle.load(f)
with open("/kaggle/input/550k-dataset/train_labels_chunked_by_family.pkl", "rb") as f:
    train_labels = pickle.load(f)
with open("/kaggle/input/550k-dataset/test_labels_chunked_by_family.pkl", "rb") as f:
    test_labels = pickle.load(f)

# Step 2: Define the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
max_sequence_length = tokenizer.model_max_length

# Step 3: Define a `compute_metrics_for_batch` function.
def compute_metrics_for_batch(sequences_batch, labels_batch, models, voting='hard'):
    # Tokenize batch
    batch_tokenized = tokenizer(sequences_batch, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
    # print("Shape of tokenized sequences:", batch_tokenized["input_ids"].shape)  # Debug print
    
    batch_dataset = Dataset.from_dict({k: v for k, v in batch_tokenized.items()})
    batch_dataset = batch_dataset.add_column("labels", labels_batch[:len(batch_dataset)])
    
    # Convert labels to numpy array of shape (1000, 1002)
    labels_array = np.array([np.pad(label, (0, 1002 - len(label)), constant_values=-100) for label in batch_dataset["labels"]])
    
    # Initialize a trainer for each model
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainers = [Trainer(model=model, data_collator=data_collator) for model in models]
    
    # Get the predictions from each model
    all_predictions = [trainer.predict(test_dataset=batch_dataset)[0] for trainer in trainers]
    
    if voting == 'hard':
        # Hard voting
        hard_predictions = [np.argmax(predictions, axis=2) for predictions in all_predictions]
        ensemble_predictions = stats.mode(hard_predictions, axis=0)[0][0]
    elif voting == 'soft':
        # Soft voting
        avg_predictions = np.mean(all_predictions, axis=0)
        ensemble_predictions = np.argmax(avg_predictions, axis=2)
    else:
        raise ValueError("Voting must be either 'hard' or 'soft'")
        
    # Use broadcasting to create 2D mask
    mask_2d = labels_array != -100
    
    # Filter true labels and predictions using the mask
    true_labels_list = [label[mask_2d[idx]] for idx, label in enumerate(labels_array)]
    true_labels = np.concatenate(true_labels_list)
    flat_predictions_list = [ensemble_predictions[idx][mask_2d[idx]] for idx in range(ensemble_predictions.shape[0])]
    flat_predictions = np.concatenate(flat_predictions_list).tolist()

    # Compute the metrics
    accuracy = accuracy_score(true_labels, flat_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, flat_predictions, average='binary')
    auc = roc_auc_score(true_labels, flat_predictions)
    mcc = matthews_corrcoef(true_labels, flat_predictions)  # Compute MCC
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc, "mcc": mcc}

# Step 4: Evaluate in batches
def evaluate_in_batches(sequences, labels, models, dataset_name, voting, batch_size=1000, print_first_n=5):
    num_batches = len(sequences) // batch_size + int(len(sequences) % batch_size != 0)
    metrics_list = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_metrics = compute_metrics_for_batch(sequences[start_idx:end_idx], labels[start_idx:end_idx], models, voting)
        
        # Print metrics for the first few batches for both train and test datasets
        if i < print_first_n:
            print(f"{dataset_name} - Batch {i+1}/{num_batches} metrics: {batch_metrics}")
        
        metrics_list.append(batch_metrics)
    
    # Average metrics over all batches
    avg_metrics = {key: np.mean([metrics[key] for metrics in metrics_list]) for key in metrics_list[0]}
    return avg_metrics

# Step 5: Load pre-trained base model and fine-tuned LoRA models
accelerator = Accelerator()
base_model_path = "facebook/esm2_t12_35M_UR50D"
base_model = AutoModelForTokenClassification.from_pretrained(base_model_path)
lora_model_paths = [
    "AmelieSchreiber/esm2_t12_35M_lora_binding_sites_cp1",
    "AmelieSchreiber/esm2_t12_35M_lora_binding_sites_v2_cp1",
]
models = [PeftModel.from_pretrained(base_model, path) for path in lora_model_paths]
models = [accelerator.prepare(model) for model in models]

# Step 6: Compute and print the metrics
test_metrics_soft = evaluate_in_batches(test_sequences, test_labels, models, "test", voting='soft')
train_metrics_soft = evaluate_in_batches(train_sequences, train_labels, models, "train", voting='soft')
test_metrics_hard = evaluate_in_batches(test_sequences, test_labels, models, "test", voting='hard')
train_metrics_hard = evaluate_in_batches(train_sequences, train_labels, models, "train", voting='hard')

print("Test metrics (soft voting):", test_metrics_soft)
print("Train metrics (soft voting):", train_metrics_soft)
print("Test metrics (hard voting):", test_metrics_hard)
print("Train metrics (hard voting):", train_metrics_hard)
