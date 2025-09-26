# ALS Proteomics Classifier Pipeline – File Explanations

This document explains the purpose of each Python file in the classifier pipeline and what parts need to be updated.
•	All sections in the code that require manual input or updates from you are marked with:
# TODO: <-- Change this
•	Please make sure to follow these TODO comments and replace the relevant lines with your own Paths to directorie or protein list.

Files :
1.	**proteins_list.py** :This file defines the final list of proteins after preprocessing and quality control. It contains 1,232 proteins selected based on the full dataset.
3.	**RFECV_age.py** : This file performs feature selection using Recursive Feature Elimination with Cross-Validation (RFECV) on the 1,232 proteins This is the first file you need to run.After running it, take the selected protein list and paste it back into the TODO section in proteins_list.py.
-	X = proteins, gender 
-	Y = age at disease onset (dichotomized patients into early-onset <60 years (label 0) and late-onset ≥60 years (label 1)
3.	**Age_classifier.py** -This file runs the main classification analysis using the selected features. It also saves evaluation metrics and plots. The 
4.	**Classification_utils.py**:  Contains all functions used by the classifier

