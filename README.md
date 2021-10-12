# UNC-NIRAL-ADHD

Will provide descriptions for each folder below:
Fall 2021 - Holds 3 subfolders. Beginning with Spring 2021 ASD Smote Playground - we conducted several experiments testing the other interpolation options for training using different settings of smote. These were also done in PyTorch as a means of learning the new deep learning library as a part of my COMP 495 contract. Models were recreated and tested against one another for different hyperparameters. SMOTE-Tomek proved to perform the best in the validation set, thus it was expanded upon to create classifiers. The performance of the classifiers was still not very good, so we revisited the interpolation of the datasets and tried to improve our interpolation model by performing a gridsearch for hyperparameters. This is the second subfolder on gridsearches. Previously the old autoencoder was a combination of two autoencoders: first one to predict 1year to 1year data. The second to predict 2year to 2year data. The resulting autoencoder encoded 2 year data and predicted 1 year information from it. The new autoencoder's gridsearch was intended for an autoencoder finding the best hyperparameters for a direct model predicting 1 year from 2 year data. Two scoring methods were used: Explained variance and negative mean squared error. The best mutual scoring is highlighted in their respective files and was used for training. The final folder is October Classifiers 2021 which held the models now using data interpolated by an autoencoder using these hyperparameters. The performance was compared against the held-out twins and its standard validation set by fold. Results did not improve.   
Models - ADHD subfolder Used for my October 2020 thesis. We used the BASC2 t-score for Hyperactivity (HYP) and Attention Problems (ATP) to indicate risk for ADHD. A subject that has a t-score > 65 (1.5 STDevs outside) is considered atypical in that behavior. Models were split into Simplified (excluding t-scores from 60-65) and contiguous (includes all data). The subfolder holds models that had their weights pre-trained by a model attempting to predict ASD risk. The models are also split by their use of interpolated data. This is created from an autoencoder predicting the missing 1 year datapoint from the 2 year datapoint. These splits total for 8 models in either folder. Taking a step back up, the ASD subfolder was used for the pre-training and then applied with the same hyperparameters to ATP Contiguous, Simplified, HYP Contiguous, Simplified.
Old Data Files - Has an unsorted list of several experiments including but not limited to: autoencoder construction by age, raw data files, PCA and T-Sne for dimensionality reduction, Scaled data, split data files by age, and classification models. These models are not particularly successful and were lumped together.
PDFs - Holds several learning contracts by semester as well as a copy of my honors thesis. Has a few articles used for understanding the research.
Raw Data Files - Holds all of the raw data needed for the project. This includes all ASD and ADHD subjects and demographics as well as their t-scores in the BASC and BRIEF. Interpolatable subjects has a list of subjects that have their 2 year but not their 1 year data. The IBIS DATA Scaling subfolder holds the a minmax scaler built on the IBIS data for which the gilmore data was scaled to.
