import pandas as pd
from honest_local import HonestEvaluator
import glob
import numpy as np

def compute_honest_score(lang, model):
    """Compute the honest score for a given language."""
    # Load results that are in the form Evals_HONEST/results_{lang}_model_iteration.csv for all iterations
    pattern = f'Evals_HONEST/results_{model}_{lang}_*.csv'
    results = glob.glob(pattern)
    
    if lang == 'ge':  # Adjust for language code
        lang = 'de'
    
    print(f"Found {len(results)} results files for {lang} and {model}")
    
    # Initialize evaluator
    evaluator = HonestEvaluator(lang)
    masked_templates = evaluator.templates(data_set="all")
    
    
    # Dictionary for honest scores
    honest_scores = {
        "gender": {gender: [] for gender in masked_templates.keys()},
        "iteration": list(range(len(results))),  # Iteration indices
    }
    
    for i, results_path in enumerate(results):
        print(f"Loading results from {results_path}")
        combined_res = pd.read_csv(results_path)
        
        
        
    for gender in masked_templates.keys():
        # Filter by gender
        df_gender = combined_res[combined_res['gender'] == gender]
        df_gender = df_gender.groupby('id').agg({'response': list, 'attribute': 'first'}).reset_index()


        # Extract attributes for the current gender
        attributes = list(set([entry['attribute'] for entry in masked_templates[gender].values()]))
       
    
        for attribute in attributes:
            # Filter by attribute
            df_gender_attribute = df_gender[df_gender['attribute'] == attribute]
            filtered_templates = {
            key: value for key, value in masked_templates[gender].items() if value['attribute'] == attribute
            }
            
            # Compute honest score
            honest_score = evaluator.honest(df_gender_attribute['response'].tolist(), filtered_templates)
            
            mean_honest_score = np.mean(honest_score)
            
            # Store the result
            honest_scores["gender"][gender].append({
                "iteration": i,  # Ensure 'i' is defined within the loop's context
                "mean_score": mean_honest_score,
                "attribute": attribute
            })
    
    # Return the final honest scores
    return honest_scores

def main():
        # Evaluate for each language
    languages = ["en-us", "ar-eg"]
    models = ["bloom-7b1", "gpt-4", "flan-t5-base"]
    print(f"Languages: {languages}")

    # Dictionary to store results
    scores = []

    # Compute scores for each language and model
    for lang in languages:
        for model in models:
            lang_scores = compute_honest_score(lang, model)
            for gender, iterations in lang_scores["gender"].items():
                for iteration_data in iterations:
                    scores.append({
                        "language": lang,
                        "model": model,
                        "gender": gender,
                        "iteration": iteration_data["iteration"],
                        "attribute": iteration_data["attribute"],
                        "mean_score": iteration_data["mean_score"]
                    })
    # to dataframe  
    df = pd.DataFrame(scores)
    # Save the scores
    df.to_csv("Evals_HONEST/scores/honest_scores.csv", index=False)

if __name__ == "__main__":
    main()