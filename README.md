# Predicting_Recipe_Ratings


# Recipe and Ratings Analysis

by Patrick Salsbury (psalsbury@ucsd.edu) & Pukhraj Falak (pfalak@ucsd.edu)

***Note***: This page explains a machine learning model built on Food.com's Recipe and Ratings dataset.


---

## Problem Identification

In this project, our goal was to create a model that accurately predicts the rating of certain recipes given information about the recipe. We decided to use the rating column because we felt like that was the most important piece of data in the entire dataset, as it was a solid incator of whether someone would want to try out the recipe or not. After merging the recipes and ratings dataset from Food.com, we had a variety of information to use inclduing but not limited to the cooking minutes, number of steps (n_steps), number of ingredients (n_ing), and nutrition info like calories, protein, soduim, etc. Since the rating column consisted of values that ranged from 1-5, we decided to use multiclass classification to predict the rating. To evaluate our model, we decided to use the accuracy score. After observing the distribution of the ratings in our dataset, we found that over 77% of the ratings given were 5-stars! This indicated that there was a major inbalance in the distribution of ratings amongst the five possible ratings. Due to this imbalance and the fact that were were performing multiclass classification, we believed that accuracy would be the best metric.

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
where baseline_columns (11 quantatitive columns) were the columns stated above. 

After processing that, the next step in our pipeline was the 'DecisionTreeClassifier'. We did not attempt to find the optimal hyperparameters, as this was just our baseline model. Our pipeline looked like this:

```py
baseline_pipeline = Pipeline([
    ('preproc', baseline_preprocessor),
    ('regressor', DecisionTreeClassifier())
])
```

After creating the pipeline, we decided to run a 'train_test_split' to get testing data for our model. After fitting the pipeline with the training data, we got the score of our model. We called '.score' with the training and testing data. The training data yieled a score of 1.0, or 100%. On the other hand, our testing score was drastically lower, achieving a score of about 60%. This suggests that we are overfitting the model on our training dataset, which could likely be caused by the fact we did not specify a max depth of for the 'DecisionTreeClassifier.' 

Realizing this, we decided to tune our model.

---

## Final Model

To start our tuning, we decided to feature engineer two new features: Date Attributes and Tags.

### Feature Engineered #1: Date Attributes

In our original dataset, we had a 'recipe_submitted_date' column which contained the date of the recipe. We decided to create three different columns from this one column: 'submitted_year', 'submitted_month', and 'submitted_dow'. We believed that the info in these columns would be more valuable since it could be generalized easier than the date. Here's an example of how the code looked like to do this:

```py
model_df['submitted_year'] = model_df['recipe_submitted_date'].dt.year
```
After performing similar code to extract the month and the data, running 'model_df.shape' would be '(219231, 26),' adding 3 new columns.


### Feature Engineered #2: Tags

Another column that we neglected in our original dataset was the tags. Since this is a list of tag variables, we are going to one-hot-encode each tag into their own columns respectively. We believed that there could be some tags that were more common in higher rating scores (ex. 'dessert' could imply a higher rating).

However, we couldn't just automatically one hot encode the 'tags' column, as it was stored as string. To solve this, we applied a specific split function to change the column to a list that contains the tags of that recipe. After doing this, we created a dataframe of one hot encoded values using all the unique tags we found as the columns. We did this by running the following: 

```py
one_hot_tags = pd.DataFrame({tag:model_df['tags'].str.contains(tag).astype(int) for tag in all_tags})
```
where 'all_tags' was a list of all unique tags.

We then merged this dataframe with our main dataframe, 'model_df'. This resulted in the shape of our data changing to (219231, 570), since there were 544 unique tags in 'tags' column and each unqiue tag had a column that contained 0s and 1s.

**Note:** We tried to do a similar approach with ingredients, but there were over 11,000 different ingredients in the dataframe. Giving each ingredient its own column seemed too excessive, so we decided to carry on to our model building.

### Decision Tree Tuning

Once we finished engineering these columns, we decided to tune our decision tree. We created a preprocessor which consisted of our baseline preprocessor and a 'OneHotEncoder' for the 'submitted_year', 'submitted_month', and 'submitted_dow' columns. We then decided to run grid search on 3 different hyperparameters: 'max_depth', 'min_samples_split', and 'criterion'. We ended up having a total of 112 different combinations of hyperparameters. After setting the amound of k-folds we want to 5, we would be testing our model on 560 different decision trees!

We identified the best hyperparameters to be '{'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 1622}'.

Now all we had to do was create a final model using this decision tree and the hyperparameters. After creating our and fitting it on the training data, we took the accuracy score for both the training set and the testing set. For the training set, received around a 75.08% score, while our testing received a 75.07% score. 

Although the accuracy of our training set decreased, our new model was better than our baseline model. We were able to eliminate the overfitting in the training, which allowed to us to receive a higher score when running the model on the test data.

---

## Fairness Analysis

Now that we have our final model trained and optimized, let's perform a fairness analysis test. To do so, we will try to determine if the model performs worse in a group X than it does for group Y. That being said, we think it would be worthwhile to conduct a fairness analysis on healthy and non-healthy groups, using the difference in RMSE ('healthy' - 'unhealthy') as the test statistic . More specifically, we can divide our data into group X(recipes with less than or equal to 1000 calories) and group Y (recipes greate than 1000 calories) in order to perform a hypothesis test:

**Null Hypothesis:** Our model is fair. The RMSE for healthy and unhealthy recipes are roughly the same, and any differences are due to random chance.

**Alternative Hypothesis:** Our model is unfair. The RMSE for unhealthy recipes is higher than the RMSE for healthier recipes.

We calculated our observed difference to be around -.0598.

<iframe src="assets/fairness_healthy.html" width=800 height=600 frameBorder=0></iframe>

We then ran a permutation test by shuffling our predictions around to simulate under the null hypothesis.

<iframe src="assets/p_val.html" width=800 height=600 frameBorder=0></iframe>

After running the permutation test, we got a p-value of 1.0, stating that we should fail to reject our null hypothesis and there isn't clear evidence from this specific test that our model is unfair.

---