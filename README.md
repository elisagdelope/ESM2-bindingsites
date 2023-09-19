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

**This model may be overfit to some extent (see below).**
Try running [this notebook](https://huggingface.co/AmelieSchreiber/esm2_t12_35M_lora_binding_sites_v2_cp3/blob/main/testing_esmb.ipynb) 
on the datasets linked to in the notebook. See if you can figure out why the metrics differ so much on the datasets. Is it due to something 
like sequence similarity in the train/test split? Is there something fundamentally flawed with the method? Splitting the sequences based on family 
in UniProt seemed to help, but perhaps a more rigorous approach is necessary? 

This model *seems* close to SOTA compared to [these SOTA structural models](https://www.biorxiv.org/content/10.1101/2023.08.11.553028v1). 
Note the especially high metrics below based on the performance on the train/test split. However, initial testing on a couple of these datasets
doesn't appear nearly as promising. If you would like to check the data preprocessing step, please see 
[this notebook](https://huggingface.co/AmelieSchreiber/esm2_t12_35M_lora_binding_sites_v2_cp3/blob/main/data_preprocessing_notebook_v1.ipynb). 

One of the primary goals in training this model is to prove the viability of using simple, single sequence only (no MSA) protein language models 
for binary token classification tasks like predicting binding and active sites of protein sequences based on sequence alone. This project 
is also an attempt to make deep learning techniques like LoRA more accessible and to showcase the competative or even superior performance 
of simple models and techniques. This however may not be as viable as other methods. The model seems to show good performance, but 
testing based on [this notebook](https://huggingface.co/AmelieSchreiber/esm2_t12_35M_lora_binding_sites_v2_cp3/blob/main/testing_esmb.ipynb) 
seems to indicate otherwise. 

The other potentially important finding is that Low Rank Adaptation (LoRA) helps dramatically improve overfitting of the models. We initially 
finetuned without LoRA and found overfitting to be a serious issue. However, after using LoRA, we found the overfitting improved quite a lot 
without any other modification. Due to the simplicity of LoRA, this may prove an important regularization technique for learning on proteins 
in the future. Keep in mind though, this did not really solve the overfitting problem despite the improvements (the finetuned model wihtout LoRA
was *very* overfit). 

Since most proteins still do not have a predicted 3D fold or backbone structure, it is useful to 
have a model that can predict binding residues from sequence alone. We also hope that this project will be helpful in this regard. 
It has been shown that pLMs like ESM-2 contain structural information in the attention maps that recapitulate the contact maps of proteins, 
and that single sequence masked language models like ESMFold can be used in atomically accurate predictions of folds, even outperforming 
AlphaFold2. In our approach we show a positive correlation between scaling the model size and data 
in a 1-to-1 fashion provides what appears to be comparable to SOTA performance, although our comparison to the SOTA models is not fair and 
comprehensive. Using the notebook linked above should help further evaluate the model, but initial findings seem pretty poor. 

This model is a finetuned version of the 35M parameter `esm2_t12_35M_UR50D` ([see here](https://huggingface.co/facebook/esm2_t12_35M_UR50D) 
and [here](https://huggingface.co/docs/transformers/model_doc/esm) for more details). The model was finetuned with LoRA for
the binay token classification task of predicting binding sites (and active sites) of protein sequences based on sequence alone. 
The model may need more training, however it still achieves better performance on the test set in terms of loss, accuracy, 
precision, recall, F1 score, ROC_AUC, and Matthews Correlation Coefficient (MCC) compared to the models trained on the smaller 
dataset [found here](https://huggingface.co/datasets/AmelieSchreiber/binding_sites_random_split_by_family) of ~209K protein sequences. Note, 
this model has a high recall, meaning it is likely to detect binding sites, but it has a precision score that is somewhat lower than the SOTA 
structural models mentioned above, meaning the model may return some false positives as well. 

## Overfitting Issues

```python
Train: ({'accuracy': 0.9908574638195745,
  'precision': 0.7748830511095647,
  'recall': 0.9862043939282111,
  'f1': 0.8678649909611492,
  'auc': 0.9886039823329382,
  'mcc': 0.8699396085712834},
Test: {'accuracy': 0.9486280975482552,
  'precision': 0.40980984516603186,
  'recall': 0.827004864790918,
  'f1': 0.5480444772577421,
  'auc': 0.890196425388581,
  'mcc': 0.560633448203768})
```
Let's analyze the train and test metrics one by one:

### **1. Accuracy**
- **Train**: 99.09%
- **Test**: 94.86%

The accuracy is notably high in both training and test datasets, indicating that the model makes correct predictions a significant 
majority of the time. The high accuracy on the test dataset signifies good generalization capabilities.

### **2. Precision**
- **Train**: 77.49%
- **Test**: 41.00%

While the precision is quite good in the training dataset, it sees a decrease in the test dataset. This suggests that a substantial 
proportion of the instances that the model predicts as positive are actually negative, which could potentially lead to a higher 
false-positive rate.

### **3. Recall**
- **Train**: 98.62%
- **Test**: 82.70%

The recall is impressive in both the training and test datasets, indicating that the model is able to identify a large proportion of 
actual positive instances correctly. A high recall in the test dataset suggests that the model maintains its sensitivity in identifying 
positive cases when generalized to unseen data.

### **4. F1-Score**
- **Train**: 86.79%
- **Test**: 54.80%

The F1-score, which is the harmonic mean of precision and recall, is good in the training dataset but sees a decrease in the test dataset. 
The decrease in the F1-score from training to testing suggests a worsened balance between precision and recall in the unseen data, 
largely due to a decrease in precision.

### **5. AUC (Area Under the ROC Curve)**
- **Train**: 98.86%
- **Test**: 89.02%

The AUC is quite high in both the training and test datasets, indicating that the model has a good capability to distinguish 
between the positive and negative classes. A high AUC in the test dataset further suggests that the model generalizes well to unseen data.

### **6. MCC (Matthews Correlation Coefficient)**
- **Train**: 86.99%
- **Test**: 56.06%

The MCC, a balanced metric which takes into account true and false positives and negatives, is good in the training set but decreases 
in the test set. This suggests a diminished quality of binary classifications on the test dataset compared to the training dataset.

### **Overall Analysis**

- **Generalization**: The metrics reveal that the model has a good generalization capability, as indicated by the high accuracy, recall, and AUC on the test dataset.
  
- **Precision-Recall Trade-off**: The model maintains a high recall but experiences a dip in precision in the test dataset, leading to a lower F1-score. It indicates a tendency to predict more false positives, which might require tuning to balance precision and recall optimally.

- **Improvement Suggestions**:
  - **Precision Improvement**: Focus on strategies to improve precision, such as feature engineering or experimenting with different classification thresholds.
  - **Hyperparameter Tuning**: Engaging in hyperparameter tuning might assist in enhancing the model's performance on unseen data.
  - **Complexity Reduction**: Consider reducing the model's complexity by training a LoRA for different weight matrices to prevent potential overfitting and improve generalization.
  - **Class Imbalance**: If the dataset has a class imbalance, techniques such as resampling or utilizing class weights might be beneficial.

So, the model performs well on the training dataset and maintains a reasonably good performance on the test dataset, 
demonstrating a good generalization capability. However, the decrease in certain metrics like precision and F1-score in the test 
dataset compared to the training dataset indicates room for improvement to optimize the model further for unseen data. It would be 
advantageous to enhance precision without significantly compromising recall to achieve a more harmonious balance between the two.

## Running Inference

You can download and run [this notebook](https://huggingface.co/AmelieSchreiber/esm2_t12_35M_lora_binding_sites_v2_cp3/blob/main/testing_and_inference.ipynb) 
to test out any of the ESMB models. Be sure to download the datasets linked to in the notebook. 
Note, if you would like to run the models on the train/test split to get the metrics, you may need to do so
locally or in a Colab Pro instance as the datasets are quite large and will not run in a standard Colab 
(you can still run inference on your own protein sequences though). 


## Training procedure

This model was finetuned with LoRA on ~549K protein sequences from the UniProt database. The dataset can be found 
[here](https://huggingface.co/datasets/AmelieSchreiber/binding_sites_random_split_by_family_550K). The model obtains 
the following test metrics, also shown above:

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
!pip install transformers -q 
!pip install peft -q
```

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