# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
## Model Details
The model is a random forest classifier from sklearn with the parameters n_estimators=200, max_depth=9.
All other parameters are set to default values. 
## Intended Use
The model is intended to us for salary prediction. 
## Training Data
The data used for training is the census.csv data provided by Udacity. 80% of the data is used for training.
## Evaluation Data
The evaluation data consists of the remaining 20% of data. 
## Metrics
The metrics for evaluation are Precision, Recall and FBeta.
The model achieves the following performance on the test set:
Precision: 0.797
Recall: 0.546
FBeta: 0.648
## Ethical Considerations
The model has not been tested against bias. Given the nature of the data, it is very likely that the model is biased and should be used with care. 
## Caveats and Recommendations
As mentioned above, the model was not tested against bias. Further, no hyperparameter tuning was performed. The model should therefore be retrained, tuned and tested properly before taken into production. 
