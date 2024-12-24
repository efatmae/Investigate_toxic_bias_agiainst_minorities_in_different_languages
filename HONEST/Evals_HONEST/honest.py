import pandas as pd

honest = pd.read_csv('scores/honest_scores.csv')
# mean_score to float and percent to float

honest['mean_score'] = honest['mean_score'].astype(float)
honest['percent'] = honest['mean_score'] *100
# group by model summing the bias scores
# Group by 'model' and 'language' and sum the 'bias' scores
# average over gender
# Step 1: Group by model, language, and gender and calculate mean

honest_grouped = honest.groupby(['model', 'language', 'gender'], as_index=False)['percent'].mean()


# Step 2: Take the mean over all three genders for each model and language
# Group again by 'model' and 'language' only, and average the gender means
#honest_grouped = honest_grouped.groupby(['model', 'language'], as_index=False)['percent'].mean() 


# Step 4: Sort the result by 'bias' in descending order
honest_grouped = honest_grouped.sort_values(by='percent', ascending=False)
print(honest_grouped)
# Save the grouped result to a CSV file
honest_grouped.to_csv('scores/honest_scores_sum.csv', index=False)
# only columns model, language and bias
