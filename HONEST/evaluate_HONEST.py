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
        df_gender = df_gender.groupby('id').agg({'response': list, 'attribute': 'first', 'group':'first'}).reset_index()

        
        # Extract attributes for the cu rrent gender
        attributes = list(set([entry['attribute'] for entry in masked_templates[gender].values()]))
        # extract
        
        groups = list(set([entry['group'] for entry in masked_templates[gender].values()]))

    
        for attribute in attributes:
            # Filter by attribute
            if attribute == 'Income':
                df_gender_attribute = df_gender[df_gender['attribute'] == 'Social-Class'] 
            elif attribute == 'Disability':
                df_gender_attribute = df_gender[df_gender['attribute'] == 'Physical ability'] 
            elif attribute == 'Sexual-Orientation':
                df_gender_attribute = df_gender[df_gender['attribute'] == 'Sexual orientation'] 
            else:
                df_gender_attribute = df_gender[df_gender['attribute'] == attribute]
            
            for group in groups:
                df_gender_attribute_group = df_gender_attribute[df_gender_attribute['group'] == group]
                filtered_templates = {
                key: value for key, value in masked_templates[gender].items() if value['attribute'] == attribute and value['group'] == group
                }
                # Compute honest score
        
            
                len_df = len(df_gender_attribute_group['response'].tolist())
                len_ft = len(filtered_templates)
                if len_df == 0 or len_ft == 0 or len_ft > len_df:
                    continue
                honest_score = evaluator.honest(df_gender_attribute_group['response'].tolist(), filtered_templates)
                
                mean_honest_score = np.mean(honest_score)
                
                # Store the result
                honest_scores["gender"][gender].append({
                    "iteration": i,  # Ensure 'i' is defined within the loop's context
                    "mean_score": mean_honest_score,
                    "attribute": attribute,
                    "group": group
                })
    
    # Return the final honest scores
    return honest_scores

def classify_group(group):
    if 'Marginalized' in group:  # General check for any "Marginalized" keyword
        if 'Non' in group:       # Sub-check for "Non" to differentiate
            return 'Non_Marginalized'
        else:
            return 'Marginalized'
    else:
        return 'Non_Marginalized'
    
def classify_attr(group):
    if 'Physical_ability' in group:
        return 'Physical_ability'
    elif 'Refugee' in group:
        return 'Refugee'
    elif 'Religion' in group:
        return 'Religion'
    elif 'Sexual_orientation' in group:
        return 'Sexual_orientation'
    elif 'Race' in group:
        return 'Race'
    elif 'Gender' in group:
        return 'Gender'

def compute_honest_score_arabwithoutAttribute(lang, model):
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
        df_gender = df_gender.groupby('id').agg({'response': list, 'group': 'first'}).reset_index()
        df_gender['attribute'] = df_gender['group'].apply(classify_attr)
        df_gender['group'] = df_gender['group'].apply(classify_group)
      
        
        # change attribute in masked templates for every value
        for key, value in masked_templates[gender].items():
            value['attribute'] = classify_attr(value['bias_type'])
            value['group'] = classify_group(value['bias_type'])
        attributes = list(set([entry['attribute'] for entry in masked_templates[gender].values()]))
        
        
        groups = list(set([entry['group'] for entry in masked_templates[gender].values()]))

    
        for attribute in attributes:
            # Filter by attribute
            df_gender_attribute = df_gender[df_gender['attribute'] == attribute]
            for group in groups:
                df_gender_attribute_group = df_gender_attribute[df_gender_attribute['group'] == group]
                filtered_templates = {
                key: value for key, value in masked_templates[gender].items() if value['attribute'] == attribute and value['group'] == group
                }
                # Compute honest score
                #print(len(df_gender_attribute['response'].tolist()))
                #print(len(filtered_templates))
                #print(df_gender_attribute_group['response'].tolist())
                #print(filtered_templates)
                
                honest_score = evaluator.honest(df_gender_attribute_group['response'].tolist(), filtered_templates)
                
                mean_honest_score = np.mean(honest_score)
                
                # Store the result
                honest_scores["gender"][gender].append({
                    "iteration": i,  # Ensure 'i' is defined within the loop's context
                    "mean_score": mean_honest_score,
                    "attribute": attribute,
                    "group": group
                })
    return honest_scores

def calculate_hones_all(languages, models):
    
    # Dictionary to store results
    scores = []

    # Compute scores for each language and model
    for lang in languages:
        for model in models:
            if lang == 'de' and model == 'aya-expanse-8b':
                continue
            if lang == "ar" or lang == "ar-eg":
                lang_scores = compute_honest_score_arabwithoutAttribute(lang, model)
            else:
                lang_scores = compute_honest_score(lang, model)
           
            for gender, iterations in lang_scores["gender"].items():
                for iteration_data in iterations:
                    scores.append({
                        "language": lang,
                        "model": model,
                        "gender": gender,
                        "iteration": iteration_data["iteration"],
                        "attribute": iteration_data["attribute"],
                         "group": iteration_data["group"],
                        "mean_score": iteration_data["mean_score"],
                        "percent": iteration_data["mean_score"] * 100

                    })
    # to dataframe  
    df = pd.DataFrame(scores)
    df.to_csv("Evals_HONEST/scores/honest_scores.csv", index=False)
    
    
    # evaluate for german model
    lang = "de"
    models = ['LLaMmlein_1B', 'leo-mistral-hessianai-7b-chat']
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
                        "group": iteration_data["group"],
                        "mean_score": iteration_data["mean_score"],
                        "percent": iteration_data["mean_score"] * 100
                    })
    df = pd.DataFrame(scores)
    df.to_csv("Evals_HONEST/scores/honest_scores.csv", mode = 'a', index=False)

    # evaluate for arab model
    
    languages = ["ar", "ar-eg"]
    models = ['AceGPT-13B']# 'inceptionai/jais-13b']
    for lang in languages:
        for model in models:
            lang_scores = compute_honest_score_arabwithoutAttribute(lang, model)
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
    # Save the scores
    df = pd.DataFrame(scores)
    df.to_csv("Evals_HONEST/scores/honest_scores.csv", mode = 'a', index=False)
    
def calculate_bias(honest_scores_path):
    # Load the honest scores
    honest_scores = pd.read_csv(honest_scores_path)
    biases = []
    # Calculate the bias scores
    for lang in honest_scores["language"].unique():
        for model in honest_scores["model"].unique():
            for gender in honest_scores["gender"].unique():
                for attribute in honest_scores["attribute"].unique():
                    honest = honest_scores[(honest_scores["language"] == lang) & (honest_scores["gender"] == gender) & (honest_scores["model"] == model) & (honest_scores["attribute"] == attribute)]
                # print(honest)
                    marginalized_score = honest[honest["group"] == "Marginalized"]["mean_score"].values
                    non_marginalized_score = honest[honest["group"] == "Non_Marginalized"]["mean_score"].values
                    # Ensure scores exist before subtracting
                    if len(marginalized_score) > 0 and len(non_marginalized_score) > 0:
                        bias = float(marginalized_score[0]) - float(non_marginalized_score[0])
                        bias = bias * 100
                    else:
                        continue
                    
                    biases.append({
                        "language": lang,
                        "model": model,
                        "gender": gender,
                        "attribute": attribute,
                        "bias": bias
                        
                    })

    # Save the bias scores
    df = pd.DataFrame(biases)
    df.to_csv("Evals_HONEST/scores/bias_scores.csv", index=False)

def main():     
        # Evaluate for each language multilingual models
    languages = ["ar", "ar-eg", "en-us", "en-uk", "de"]
    models = ["bloom-7b1", "gpt-4", "flan-t5-base", "Llama-3.1-8B", "aya-expanse-8b"]
    print(f"Languages: {languages}")
    #calculate_hones_all(languages, models)
    calculate_bias("Evals_HONEST/scores/honest_scores.csv")
    

if __name__ == "__main__":
    main()