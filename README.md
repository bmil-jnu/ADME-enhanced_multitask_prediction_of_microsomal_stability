# Cross-species multi-task learning with molecular and ADME descriptors for liver microsomal metabolic stability
Authors: Subhin Seomun, Sunyong Yoo

## Description
This project introduces a species-aware, multi-task, multi-modal framework for predicting liver microsomal metabolic stability across human (HLM), rat (RLM), and mouse (MLM). Our model fuses SMILES-derived fingerprints, graph-neural features, and in-silico ADME/physicochemical descriptors to improve cross-species generalization and to explain model decisions. Interpretability combines descriptor-level SHAP  with EdgeSHAPer, plus fragment–ADME enrichment to link local chemistries to system-level properties. The repository contains:
- **[Dataset](Dataset)**: The dataset used in the paper.
- **[Model](Model)**:  The implementation of the ADME-enhanced multitask prediction of microsomal stability prediction algorithm.
- **[SHAP](Analysis/SHAP)**: Code and analysis results generated with SHAP.
- **[Edgeshaper](Analysis/Edgeshaper)**: Code and analysis results generated with Edgeshaper.
- **[Heatmap](Analysis/SHAP)**: Code and analysis results generated with Heatmap.


## Dependency
The project has the following dependencies:

- 'Python==3.12.5'
- 'PyTorch==2.2.0'
- 'Torch_Geometric==2.6.1'
- 'scikit-learn==1.6.1'

## Contacts
For questions, suggestions, or feedback, please contact:
- Email: [munsu931122@jnu.ac.kr]
- Email: [syyoo@jnu.ac.kr]
