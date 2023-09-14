---
library_name: peft
license: mit
datasets:
- AmelieSchreiber/binding_sites_random_split_by_family_550K
language:
- en
metrics:
- accuracy
- precision
- recall
- f1
- roc_auc
- matthews_correlation
pipeline_tag: token-classification
tags:
- ESM-2
- biology
- protein language model
- binding sites
---
# ESM-2 for Binding Site Prediction

This model is a finetuned version of the 35M parameter `esm2_t12_35M_UR50D` ([see here](https://huggingface.co/facebook/esm2_t12_35M_UR50D) 
and [here](https://huggingface.co/docs/transformers/model_doc/esm) for more details). The model was finetuned with LoRA for
the binay token classification task of predicting binding sites (and active sites) of protein sequences based on sequence alone. 
The model may be underfit and undertrained, however it still achieved better performance on the test set in terms of loss, accuracy, 
precision, recall, F1 score, ROC_AUC, and Matthews Correlation Coefficient (MCC) compared to the models trained on the smaller 
dataset [found here](https://huggingface.co/datasets/AmelieSchreiber/binding_sites_random_split_by_family) of ~209K protein sequences. Note, 
this model has a high recall, meaning it is likely to detect binding sites, but it has a low precision, meaning the model will likely return 
false positives as well. 

## Training procedure

This model was finetuned on ~549K protein sequences from the UniProt database. The dataset can be found 
[here](https://huggingface.co/datasets/AmelieSchreiber/binding_sites_random_split_by_family_550K). The model obtains 
the following test metrics:

```python
Epoch: 3
Training Loss: 0.029100
Validation Loss: 0.291670
Accuracy: 0.948626
Precision: 0.409795
Recall: 0.826979
F1: 0.548025
Auc: 0.890183
Mcc: 0.560612
```

### Framework versions

- PEFT 0.5.0

## Using the model

To use the model on one of your protein sequences try running the following:

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from peft import PeftModel
import torch

# Path to the saved LoRA model
model_path = "AmelieSchreiber/esm2_t12_35M_lora_binding_sites_v2_cp3"
# ESM2 base model
base_model_path = "facebook/esm2_t12_35M_UR50D"

# Load the model
base_model = AutoModelForTokenClassification.from_pretrained(base_model_path)
loaded_model = PeftModel.from_pretrained(base_model, model_path)

# Ensure the model is in evaluation mode
loaded_model.eval()

# Load the tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Protein sequence for inference
protein_sequence = "MAVPETRPNHTIYINNLNEKIKKDELKKSLHAIFSRFGQILDILVSRSLKMRGQAFVIFKEVSSATNALRSMQGFPFYDKPMRIQYAKTDSDIIAKMKGT"  # Replace with your actual sequence

# Tokenize the sequence
inputs = loaded_tokenizer(protein_sequence, return_tensors="pt", truncation=True, max_length=1024, padding='max_length')

# Run the model
with torch.no_grad():
    logits = loaded_model(**inputs).logits

# Get predictions
tokens = loaded_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Convert input ids back to tokens
predictions = torch.argmax(logits, dim=2)

# Define labels
id2label = {
    0: "No binding site",
    1: "Binding site"
}

# Print the predicted labels for each token
for token, prediction in zip(tokens, predictions[0].numpy()):
    if token not in ['<pad>', '<cls>', '<eos>']:
        print((token, id2label[prediction]))
```