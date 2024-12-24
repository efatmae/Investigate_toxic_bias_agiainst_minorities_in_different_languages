import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Define output directory
output_dir = "Evals_HONEST/scores/plots"
os.makedirs(output_dir, exist_ok=True)

# Define a continuous color map and marker styles for the APIs
color_map = plt.get_cmap('viridis')  # A continuous colormap
markers = ['o', 's', '^', '*']  # Different markers for each model


def canonical_attribute(attr: str) -> str:
    # Normalize by replacing underscores and hyphens with spaces
    normalized = attr.replace('_', ' ').replace('-', ' ')
    # Convert to lowercase for consistent lookup
    normalized_lower = normalized.lower().strip()
    
    # Define a mapping from normalized forms to their canonical forms
    # Keys should be the fully normalized lowercase version
    canonical_map = {
        'religion': 'Religion',
        'physical ability': 'Physical ability',
        'race': 'Race',
        'sexual orientation': 'Sexual orientation',
        'gender': 'Gender',
        'refugee': 'Refugee',
        'refugees': 'Refugee',  # maps plural to singular
        'social class': 'Social class',
        'disability': 'Physical ability',
        'income': 'Social class'
    }

    return canonical_map.get(normalized_lower, normalized.title())

def plot_honest_differences(data, x_axis_limits):
    unique_attrs = data['attribute'].unique()
    
    groups = set(map(canonical_attribute, unique_attrs))
    
    attr_positions = {attr: idx for idx, attr in enumerate(groups)}
    print(groups)
    genders = data['gender'].unique()

    # Define attribute colors
    steps = np.linspace(0, 1, len(groups))
    attribute_colors = {group: color_map(step) for group, step in zip(groups, steps)}

    # We'll define a different color scheme for languages
    languages = data['language'].unique()
    lang_steps = np.linspace(0, 1, len(languages))
    lang_colors = {lang: plt.cm.tab20(step) for lang, step in zip(languages, lang_steps)}

    for gender in genders:
        gender_data = data[data['gender'] == gender]
        models = gender_data['model'].unique()

       
        main_model = sorted(models)[0] if len(models) > 0 else None

        fig, ax = plt.subplots(figsize=(12, 8))
        
        
        for i, (model, marker) in enumerate(zip(models, markers)):
            model_data = gender_data[gender_data['model'] == model]

            for _, row in model_data.iterrows():
                if pd.notna(row['bias']) and pd.notna(row['attribute']) and pd.notna(row['language']):
                    attr = canonical_attribute(row['attribute'])
                    lang = row['language']
                    # Decide on colors by language now
                    color = lang_colors[lang]
                    
                    # vertical shift by model index
                    y_pos = attr_positions[attr] + i * 0.1
                    ax.scatter(
                        row['bias'],
                        y_pos,
                        color=color,
                        marker=marker,
                        edgecolor='black'
                        # no label here; we'll create a custom legend below
                    )

        # Draw a vertical line at x=0
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

        # Set the ticks and labels for the y-axis
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=14)
        ax.set_ylim(-0.5, len(groups)-0.5)

        # X-axis and title
        plt.xlabel('HONEST Score Differences\n(Marginalized - Non-Marginalized)', fontsize=16, fontweight='bold')
        plt.ylabel('Attribute', fontsize=16, fontweight='bold')
        plt.title(f'HONEST Score Differences for Gender: {gender}', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=14)
        plt.xlim(x_axis_limits)

        # Create a custom legend: one entry per language using the main model
        # If main_model is None (no data), skip legend
        if main_model is not None:
            # Use the first model's marker (markers[0]) or the chosen main_model's corresponding marker
            # We'll assume the first marker for simplicity:
            main_marker = markers[0]

            legend_handles = []
            for lang in languages:
                handle = ax.scatter([], [], 
                                    color=lang_colors[lang], 
                                    marker=main_marker, 
                                    edgecolor='black', 
                                    label=f"{main_model} ({lang})")
                legend_handles.append(handle)
            ax.legend(handles=legend_handles, fontsize=12, loc='upper right')
        
        # Save the plot
        output_file_path = os.path.join(output_dir, f'honest_score_differences_{gender}.png')
        plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
        plt.close()

def plot_Honest_marg_nonmar(data):
    unique_attrs = data['attribute'].unique()
    
    groups = set(map(canonical_attribute, unique_attrs))
    
    attr_positions = {attr: idx for idx, attr in enumerate(groups)}

    # Define attribute colors
    steps = np.linspace(0, 1, len(groups))
    attribute_colors = {group: color_map(step) for group, step in zip(groups, steps)}

    languages = data['language'].unique()
    for lang in languages:
        lang_data = data[data['gender'] == lang]
        # group by model averaging the honest scores and saving the variance
        lang_data = lang_data.groupby(['model'], as_index=False)['percent'].mean()
        lang_data = lang_data.sort_values(by='percent', ascending=False)
        var = lang_data.groupby(['model'], as_index=False)['percent'].var()

       
        main_model = sorted(models)[0] if len(models) > 0 else None

        fig, ax = plt.subplots(figsize=(12, 8))
        
        
        for i, (model, marker) in enumerate(zip(models, markers)):
            model_data = gender_data[gender_data['model'] == model]

            for _, row in model_data.iterrows():
                if pd.notna(row['bias']) and pd.notna(row['attribute']) and pd.notna(row['language']):
                    attr = canonical_attribute(row['attribute'])
                    lang = row['language']
                    # Decide on colors by language now
                    color = lang_colors[lang]
                    
                    # vertical shift by model index
                    y_pos = attr_positions[attr] + i * 0.1
                    ax.scatter(
                        row['bias'],
                        y_pos,
                        color=color,
                        marker=marker,
                        edgecolor='black'
                        # no label here; we'll create a custom legend below
                    )



# Apply the plotting function to the data
df = pd.read_csv("Evals_HONEST/scores/bias_scores.csv")
plot_honest_differences(df, x_axis_limits=(-0.2, 0.2))

df = pd.read_csv("Evals_HONEST/scores/honest_scores.csv")
plot_Honest_marg_nonmar(df)
