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
## Training procedure

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