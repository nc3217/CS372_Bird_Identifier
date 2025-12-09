# CS372_Bird_Identifier

## What it Does

This project aims to assist bird watchers and researchers in identifying bird species using image classification. By inputting an image of one of the 1486 bird species that this model is trained on, this model will be able to classify the bird species. This model can be implemented into field cameras that researchers use in fieldwork to allow automatic classification of birds for research purposes instead of relying on human identification. This project compared both a CLIP pretrained model and an EfficientNet model, with the EfficientNet model yielding the best outcomes.

## Quick Start

To set up this project, create a python environment. Then, install python dependencies as indicated in the SETUP.md. 

## Video Links



## Evaluation

INFERENCE TIMING SUMMARY:
  Total images: 41608, Time: 327.392s
  Throughput: 127.09 img/s
  Mean latency: 4.84 ms
  Median latency: 4.73 ms
  p95 latency: 5.44 ms

METRICS (TEST SET):
  Top-1 Accuracy : 0.5685
  Macro F1       : 0.5650
  Precision      : 0.5985
  Recall         : 0.5685
  Top-5 Accuracy : 0.8024


EXAMPLE MISCLASSIFICATION
Image: /kaggle/input/inatbirds100k/birds_train_small/04408_Animalia_Chordata_Aves_Pelecaniformes_Threskiornithidae_Plegadis_chihi/467d1d79-aa4c-4a3f-8ae8-4e537f34cf07.jpg
TRUE LABEL:      04408_Animalia_Chordata_Aves_Pelecaniformes_Threskiornithidae_Plegadis_chihi
PREDICTED LABEL: 04576_Animalia_Chordata_Aves_Suliformes_Phalacrocoracidae_Phalacrocorax_capensis
Top-5 Predictions:
   04576_Animalia_Chordata_Aves_Suliformes_Phalacrocoracidae_Phalacrocorax_capensis 0.1296
   04408_Animalia_Chordata_Aves_Pelecaniformes_Threskiornithidae_Plegadis_chihi 0.1220
   04409_Animalia_Chordata_Aves_Pelecaniformes_Threskiornithidae_Plegadis_falcinellus 0.1144
   03647_Animalia_Chordata_Aves_Gruiformes_Gruidae_Anthropoides_virgo 0.0643
   03646_Animalia_Chordata_Aves_Gruiformes_Gruidae_Anthropoides_paradiseus 0.0625


<img width="1024" height="685" alt="image" src="https://github.com/user-attachments/assets/05e3acad-1d13-47ce-ae66-afb993e473b7" />
🔎 Top Predictions:
1. 03838_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Spinus_tristis — 0.9943
2. 03809_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Carduelis_carduelis — 0.0017
3. 03812_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Chloris_chloris — 0.0008
4. 03836_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Spinus_psaltria — 0.0007
5. 03815_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Coccothraustes_vespertinus — 0.0006


## Individual Contributions

Irene: Irene worked on much of the baseline modeling. She also worked on the fine-tuning for the EfficientNet model.

Neeharika: Neeharika worked on the CLIP logistic regression baseline model, and she also worked on the inference/testng for the EfficientNet model.



