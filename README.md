# Assignment A03

The purpose of this assignment is to learn and implement various type of sampling methods in Python. More specifically, it focuses on Majority Undersampling, Minority Oversampling, and SMOTE techniques. All of these methods were done using the imblearn library. 

This project used a training set version of the California Housing dataset within the sklearn.datasets package in Python provided to us by our professor. 

In order to run this script, you must do the following:

1. Clone the Respository in GitHub Desktop.
2. Open the repository in Visual Studio Code and make sure Python 3.12 (or similar) is set as the virtual environment.
3. Install the required packages by doing: python install -r requirements.txt
4. Run the A03.py script by doing: python src/A03.py

After doing this, you should see a series of seven figures in the figs folder. 

## Comments on Analyses: 

The first histogram made, histogram_MHV.png, shows us the median home values within our dataset. This histogram was made to help determine a threshold for making our response variable categorical. 

### Majority Undersampling: 

The Majoirty Undersampling performed fairly well under the decision tree model. It did not seem to completely memorize the training data as we typically expect from decision trees. In terms of its model accuracy with the testing set, it had a moderately high accuracy of 83% and weighted average F1 score of 86%. When we look at the confusion matrix, we see this model tended to favor false negatives over false postives. Overall, I think this model performed well, but may need more tuning to really fit our data better. 

### Minority Oversampling: 

The Minority Oversampling model performed significantly better than the Majority Undersampling model. It did memorize the training set, which we expect out of decision tree analyses. But, its predictive ability was significantly better than the oversampling model. It had an overall accuracy of about 92%, and a weighted average F1 score of about 92% as well. The confusion matrix of the testing dataset showed about an even split in false postives and false negatives, which is what we look for in models. Overall, this model ran really well with our data. 

### SMOTE: 

Finally, the SMOTE model seemed to perform slightly better than undersampling but not as well as oversampling. It did not memorize our training set as we would expect it to, similar to undersampling. It had an overall accuracy of about 89%, and a weight average F1 score of about 90%. Both values are better than our undersampling model, but not quite as good as the oversampling model. The confusion matrix shows behavors similar to the undersampling model. SMOTE seems to favor false negative when predicting over false postives. Overall, this model ran well but would probably need some slight tuning to perform better than oversampling did. 

### Comparing All Three Models: 

Overall, all three sampling models performed fairly well under a decision tree approach. The undersampling and the SMOTE models seemed to perform fairly similarily, with a common favoring in false negatives in the model. They both also didn't memorize the training set. This is not a requirement for being a good model, but it goes against what we expect from decision tree models. The oversampling model performed better than the undersampling and SMOTE models. It had higher accuracy and F1 scores, and did not favor false negative or false postives over the other one. Overall, this is the best model of the three. Since this oversampling performed the best, this model was used for the replication of models later. 

### Reproducibility: Oversampling Model

#### 30 Trials: 

First we performed replication of the oversampling model with 30 trials. We kept the same decision tree and oversampling model as we did when ran the original model for consistency. After generating 30 random training and testing splits, and running the model, we plotted a histogram the Accuracy (Accuracy30.png), Precision (Precision30.png), and Recall (Recall30.png) of the all 30 trials. All three histograms seem be approximately bell shaped as desired. Precision seemed to have some folds that performed a little worse than others, but overall, all three metrics show high accuracy, precision, and recall values. 

#### 100 Trials:

Finally, we performed the replication of the oversampling with 100 trials. We kept every model the same as we did with the 30 trials. The increase in the trials definitely improves the model. The historgams of the Accuracy (Accuracy100.png), Precision (Precision100.png), and Recall (Recall100.png) of the all 100 trials all become more bell shaped and symmetric, compared to the 30 trials. The accuracy values seemed to increase slightly with more trials, and precision and recall seemed to perform similarly to the 30 trials. Overall, the oversampling model still performed well when reproduced many times. 
