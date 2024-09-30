The objective of this challenge is to design and build a predictive model capable of accurately determining the probability of an individual having heart disease.

**EVALUATION**

The error metric for this competition is Accuracy.
However, we will also focus on **precision and recall** of the model predictions by taking some trade-off between them. Also finding the threshold for categorising the predictions probability in either 0:No or 1:Yes.


**NOTE:**

Let keep it in mind that one of the best evaluation metric will be the  ROC-AUC score and the F1 score.    

## VISUALS


The target that we will be predicting is inbalanced with 0 and 1 having about 20% and 80% respectively. 

![png](output_10_0.png)
       
![png](output_11_0.png)
        
![png](output_12_0.png)
    


Patients having chest pain of category 0 have high chance of heart disease than other category.

### EXERCISE INDUCED ANGINA (exang)
    
![png](output_15_0.png)
    


Question:

What category of chest pain do patients with exercise induced anginia have?.


```python
exang1 = train[train['exang']==1]
exang0 = train[train['exang']==0]


    
![png](output_17_0.png)
    
   
![png](output_18_0.png)
       
![png](output_19_0.png)
    


**INSIGHTS**

* Individuals with *Exercise induced Angina* have high chance of heart disease.
* Individuals with *chest pain (0)* have heart disease.
*  

## Preprocessing.

* Transform categorical features with more than two unique entries with onehot.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Age</th>
      <th>Sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>...</th>
      <th>slope_2</th>
      <th>ca_0</th>
      <th>ca_1</th>
      <th>ca_2</th>
      <th>ca_3</th>
      <th>ca_4</th>
      <th>thal_0</th>
      <th>thal_1</th>
      <th>thal_2</th>
      <th>thal_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16167</td>
      <td>33</td>
      <td>0</td>
      <td>1</td>
      <td>158</td>
      <td>205</td>
      <td>1</td>
      <td>0</td>
      <td>154</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11275</td>
      <td>53</td>
      <td>1</td>
      <td>2</td>
      <td>198</td>
      <td>154</td>
      <td>0</td>
      <td>1</td>
      <td>104</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13251</td>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>101</td>
      <td>202</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19921</td>
      <td>75</td>
      <td>0</td>
      <td>0</td>
      <td>113</td>
      <td>306</td>
      <td>1</td>
      <td>2</td>
      <td>88</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11293</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>139</td>
      <td>419</td>
      <td>1</td>
      <td>1</td>
      <td>166</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>

    target
    1    0.813501
    0    0.186499
    Name: proportion, dtype: float64



**The model may have confusion rate of about 0.19 for observations that doesnot fall under the following:
Exercise Induced Anginia: No (0)
Chest Pain categories excludiing zero(0).**

# Modelling.
## Logistic Regression.

    Train ROC-AUC SCORE: 0.8270800873211603
    Test ROC-AUC SCORE: 0.8376050420168069
                  precision    recall  f1-score   support
    
               0       0.54      0.84      0.66       136
               1       0.96      0.84      0.89       595
    
        accuracy                           0.84       731
       macro avg       0.75      0.84      0.78       731
    weighted avg       0.88      0.84      0.85       731
    
      
![png](output_33_0.png)
    
    
![png](output_34_0.png)
    


## RandomForest.


    Train ROC-AUC SCORE: 0.9070821196070424
    Validation ROC-AUC SCORE: 0.598109243697479
                  precision    recall  f1-score   support
    
               0       0.52      0.25      0.34       136
               1       0.85      0.95      0.89       595
    
        accuracy                           0.82       731
       macro avg       0.68      0.60      0.62       731
    weighted avg       0.78      0.82      0.79       731
    
    


## RandomForest Feature Importances
    
![png](output_38_0.png)



    
![png](output_39_0.png)
    


    tuned_model.best_threshold_=0.78
    tuned_model.best_score_=0.88
    


    
![png](output_44_1.png)
    


### plot threshold details.
    
![png](output_46_0.png)
    

    
![png](output_48_1.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>exang</th>
      <th>cp</th>
      <th>thalach</th>
      <th>target</th>
      <th>proba</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16501</td>
      <td>1</td>
      <td>0</td>
      <td>170</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10444</td>
      <td>1</td>
      <td>0</td>
      <td>74</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14288</td>
      <td>1</td>
      <td>0</td>
      <td>73</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10409</td>
      <td>1</td>
      <td>1</td>
      <td>192</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17330</td>
      <td>0</td>
      <td>3</td>
      <td>122</td>
      <td>1</td>
      <td>0.537527</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2692</th>
      <td>14964</td>
      <td>0</td>
      <td>3</td>
      <td>163</td>
      <td>1</td>
      <td>0.538947</td>
    </tr>
    <tr>
      <th>2693</th>
      <td>16774</td>
      <td>1</td>
      <td>1</td>
      <td>95</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2694</th>
      <td>18884</td>
      <td>1</td>
      <td>0</td>
      <td>170</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2695</th>
      <td>10000</td>
      <td>1</td>
      <td>0</td>
      <td>147</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2696</th>
      <td>17660</td>
      <td>0</td>
      <td>3</td>
      <td>91</td>
      <td>0</td>
      <td>0.488921</td>
    </tr>
  </tbody>
</table>
<p>2697 rows × 6 columns</p>
</div>



**CONCLUSION.**

The most important factors of *Heart Disease* exercise induced angina(exang), chest pain(cp), 
and maximum heart rate achieved(thalach).



**THANK YOU.**
