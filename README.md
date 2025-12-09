# CS372_Bird_Identifier

## What it Does

This project aims to assist bird watchers and researchers in identifying bird species using image classification. By inputting an image of one of the 1486 bird species that this model is trained on, this model will be able to classify the bird species. This model can be implemented into field cameras that researchers use in fieldwork to allow automatic classification of birds for research purposes instead of relying on human identification. This project compared both a CLIP pretrained model and an EfficientNet model, with the EfficientNet model yielding the best outcomes.

## Quick Start

To set up this project, create a Python environment. Then, install Python dependencies as indicated in the SETUP.md. You can then use the full notebooks in the notebooks folder, or you can use train.py and evaluate.py in the src folder.

## Video Links



## Evaluation

INFERENCE TIMING SUMMARY:
- Total images: 41608, Time: 327.392s
- Throughput: 127.09 img/s
- Mean latency: 4.84 ms
- Median latency: 4.73 ms
- p95 latency: 5.44 ms

METRICS (TEST SET):
- Top-1 Accuracy : 0.5685
- Macro F1       : 0.5650
- Precision      : 0.5985
- Recall         : 0.5685
- Top-5 Accuracy : 0.8024

Example Correct Classification
Input image: Goldfinch
<img width="1024" height="685" alt="image" src="https://github.com/user-attachments/assets/05e3acad-1d13-47ce-ae66-afb993e473b7" />
🔎 Top-5 Predictions:
1. 03838_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Spinus_tristis — 0.9943
2. 03809_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Carduelis_carduelis — 0.0017
3. 03812_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Chloris_chloris — 0.0008
4. 03836_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Spinus_psaltria — 0.0007
5. 03815_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Coccothraustes_vespertinus — 0.0006


Example Misclassification<br>
<img width="275" height="273" alt="Screenshot 2025-12-09 at 2 53 20 PM" src="https://github.com/user-attachments/assets/1b80fbfc-6e3e-4ab6-a452-ccb67fc5f010" />
<br>TRUE LABEL:      03771_Animalia_Chordata_Aves_Passeriformes_Corvidae_Nucifraga_columbiana
<br>PREDICTED LABEL: 03113_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Accipiter_gentilis
<br>Top-5 Predictions:
1. 03113_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Accipiter_gentilis 0.4891
2. 03115_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Accipiter_striatus 0.1458
3. 03114_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Accipiter_nisus 0.1039
4. 03130_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Buteo_plagiatus 0.0567
5. 04272_Animalia_Chordata_Aves_Passeriformes_Turdidae_Myadestes_townsendi 0.0387



## Individual Contributions

Irene: Irene worked on loading and splitting the dataset, the random baseline model, fine-tuning the EfficientNet model, the evaluation metrics computations, and the image prediction function.

Neeharika: Neeharika worked on the CLIP logistic regression baseline model, the initial EfficientNet model, evaluating the EfficientNet model on the test data, and inference timing. 



