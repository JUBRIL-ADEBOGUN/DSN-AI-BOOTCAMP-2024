The objective of this challenge is to design and build a predictive model capable of accurately determining the probability of an individual having heart disease.

**EVALUATION**

The error metric for this competition is Accuracy.
However, we will also focus on **precision and recall** of the model predictions by taking some trade-off between them. Also finding the threshold for categorising the predictions probability in either 0:No or 1:Yes.


**NOTE:**

Let keep it in mind that one of the best evaluation metric will be the  ROC-AUC score and the F1 score.    

## EXPLORATORY ANALYSIS.


The target that we will be predicting is inbalanced with 0 and 1 having about 20% and 80% respectively. 
   


Patients having chest pain of category 0 have high chance of heart disease than other category.

### EXERCISE INDUCED ANGINA (exang)
Question:

What category of chest pain do patients with exercise induced anginia have?.


**INSIGHTS**

* Individuals with *Exercise induced Angina* have high chance of heart disease.
* Individuals with *chest pain (0)* have heart disease.


**The model may have confusion rate of about 0.19 for observations that doesnot fall under the following:
Exercise Induced Anginia: No (0)
Chest Pain categories excludiing zero(0).**

# MODELLING RESULTS.

## Logistic Regression.

    Train ROC-AUC SCORE: 0.8270800873211603
    Test ROC-AUC SCORE: 0.8376050420168069
                  precision    recall  f1-score   support
    
               0       0.54      0.84      0.66       136
               1       0.96      0.84      0.89       595
    
        accuracy                           0.84       731
       macro avg       0.75      0.84      0.78       731
    weighted avg       0.88      0.84      0.85       731
    

## RandomForest.


    Train ROC-AUC SCORE: 0.9070821196070424
    Validation ROC-AUC SCORE: 0.598109243697479
                  precision    recall  f1-score   support
    
               0       0.52      0.25      0.34       136
               1       0.85      0.95      0.89       595
    
        accuracy                           0.82       731
       macro avg       0.68      0.60      0.62       731
    weighted avg       0.78      0.82      0.79       731
    
    tuned_model.best_threshold_=0.78
    tuned_model.best_score_=0.88


**CONCLUSION.**

The most important factors of *Heart Disease* exercise induced angina(exang), chest pain(cp), 
and maximum heart rate achieved(thalach).



**THANK YOU.**
