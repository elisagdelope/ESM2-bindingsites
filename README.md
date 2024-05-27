
# ESM-2 for Binding Site Prediction 

Using ESM-2 protein language model to predict binding sites of proteins from their sequence alone. I use the 6 layer, 8M parameter [esm2_t6_8M_UR50D](https://github.com/facebookresearch/esm), trained on [this dataset](https://huggingface.co/datasets/AmelieSchreiber/general_binding_sites). [See here](https://huggingface.co/facebook/esm2_t6_8M_UR50D) and [here](https://huggingface.co/docs/transformers/model_doc/esm) for more details on the pre-trained model.  

The goal is to use single sequence only (no MSA) protein language models for binary token classification tasks like predicting binding and active sites of protein sequences based on sequence alone. The model was finetuned with and without LoRA for the binary token classification task of predicting binding sites (and active sites) of protein sequences based on sequence alone. 

On data (protein sequence) pre-processing: [check notebook](https://github.com/elisagdelope/ESM2-bindingsites/blob/master/data_preprocessing_notebook_v1.ipynb)



## Acknowledgements
A great deal of heavy work with ESM2 models has been done by [Amelie Schreiber on hugging face](https://huggingface.co/AmelieSchreiber), whose blog posts (and code) have inspired this repository. My contribution is integrating these models with the [DeepChem](https://github.com/deepchem/deepchem/tree/master) open source framework. This work was sponsored by the [Google Summer of Code](https://summerofcode.withgoogle.com/) program.


