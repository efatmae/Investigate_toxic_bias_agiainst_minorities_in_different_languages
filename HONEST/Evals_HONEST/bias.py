import pandas as pd

bias = pd.read_csv('scores/bias_scores.csv')
# group by model summing the bias scores
# Group by 'model' and 'language' and sum the 'bias' scores
# average over gender
# Step 1: Group by model, language, and gender and calculate mean
bias_grouped = bias.groupby(['model', 'language', 'attribute'], as_index=False)['bias'].mean()

# Step 2: Take the mean over all three genders for each model and language
# Group again by 'model' and 'language' only, and average the gender means
#bias_grouped = bias_grouped.groupby(['model', 'language'], as_index=False)['bias'].mean()



# Step 4: Sort the result by 'bias' in descending order
bias_grouped = bias_grouped.sort_values(by='bias', ascending=False)

# Save the grouped result to a CSV file
bias_grouped.to_csv('scores/bias_scores_sum.csv', index=False)
# only columns model, language and bias
