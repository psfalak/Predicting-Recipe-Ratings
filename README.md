# Predicting_Recipe_Ratings


# Recipe and Ratings Analysis

by Patrick Salsbury (psalsbury@ucsd.edu) & Pukhraj Falak (pfalak@ucsd.edu)

***Note***: This page explains a machine learning model built on Food.com's Recipe and Ratings dataset.


---

## Problem Identification

In this project, our goal was to create a model that accurately predicts the rating of certain recipes given information about the recipe. We decided to use the rating column because we felt like that was the most important piece of data in the entire dataset, as it was a solid incator of whether someone would want to try out the recipe or not. After merging the recipes and ratings dataset from Food.com, we had a variety of information to use inclduing but not limited to the cooking minutes, number of steps (n_steps), number of ingredients (n_ing), and nutrition info like calories, protein, soduim, etc. Since the rating column consisted of values that ranged from 1-5, we decided to use multiclass classification to predict the rating. To evualate our model, we decided to use the F1-score. After observing the distribution of the ratings in our dataset, we found that over 77% of the ratings given were 5-stars! This indicated that there was a major inbalance in the distribution of ratings amongst the five possible ratings. Since F1 score accounts for this class imbalance, we decided to go with that. 




---

## Baseline Model

Our baseline model pipeline consisted of a 'preprocessor' and a 'DecisionTreeClassifier'. In the preprocessor, We decided use the 'StandardScaler()' feature on the following columns:

* 'cooking_minutes'
* 'n_steps', 
* 'n_ingredients'
* 'n_tags'
* 'calories' 
* 'total_fat'
* 'protein'
* 'sugar_data'
* 'sodium'
* 'sat_fat'
* 'carbs'

We believed that purely looking at the information about the recipe could give us solid start to predicting the rating of a specific recipe. We embedded this in a column transformer before we inputted it into our pipeline. The code was as follows:

```py
baseline_preprocessor = ColumnTransformer(
    transformers = [
        ('std_scaler', StandardScaler(), baseline_columns)
    ]
)
```
where baseline_columns were the columns stated above.

After processing that, the next step in our pipeline was the 'DecisionTreeClassifier'. We did not attempt to find the optimal hyperparamets, as this was just our baseline model. Our pipeline looked like this:

```py
baseline_pipeline = Pipeline([
    ('preproc', baseline_preprocessor),
    ('regressor', DecisionTreeClassifier())
])
```


---

## Final Model

Text for the final model

---

## Fairness Analysis

This is our fairness analysis

---
