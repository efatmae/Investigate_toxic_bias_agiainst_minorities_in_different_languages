import pandas as pd

def split_by_group(file_path, column_name, output_file_path):
    """
    Splits a column in a CSV file based on 'Marginalized' or 'Non_Marginalized'
    into two new columns: 'attribute' and 'group'.
    
    Parameters:
    - file_path: str, path to the input CSV file
    - column_name: str, name of the column to split
    - output_file_path: str, path to save the updated CSV file
    
    Returns:
    - DataFrame with new columns
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Identify the group and attribute by splitting on "Marginalized" or "Non_Marginalized"
    data['group'] = data[column_name].str.extract('(Marginalized|Non_marginalized)')
    data['attribute'] = data[column_name].str.replace('(Marginalized|Non_marginalized)', '', regex=True).str.strip('_')
    
    # Save the updated dataset
    data.to_csv(output_file_path, index=False)
    
    print(f"Updated file saved to: {output_file_path}")
    return data

# Example usage
file_path = "results_AceGPT-13B_ar_1.csv"  # Replace with your file path
column_name = "group"  # Replace with your column name
output_file_path = "results_AceGPT-13B_ar_1_new.csv"  # Replace with the desired output file path

updated_data = split_by_group(file_path, column_name, output_file_path)

# Display the updated DataFrame
print(updated_data.head())
