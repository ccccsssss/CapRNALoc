# CapRNALoc
Descriptions
./model: This folder contains two pre-trained models, corresponding to the balanced and imbalanced scenarios, respectively.
./oir_data: This folder contains three original datasets. The balanced dataset is from iRNAac4C, and the imbalanced datasets are from Meta-ac4C.
CGR_datasets.zip: This compressed archive contains three benchmark datasets that have been feature-encoded using the FCGR (Frequency Chaos Game Representation) method.
CGR_encoding.py: Python script for FCGR feature encoding.
evaluation.py: Python script for performing 4-fold cross-validation.
model.py: Python script for the CapsBAM prediction model.
train.py: Python script for training the model.
test.py: Python script for evaluating the model on an independent test set.
