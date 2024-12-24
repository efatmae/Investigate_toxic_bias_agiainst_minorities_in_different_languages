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
    data['group'] = data[column_name].apply(lambda x: "Marginalized" if "Marginalized" in x else "Non_Marginalized")
    data['attribute'] = data[column_name].str.replace("Marginalized", "").str.replace("Non_marginalized", "").str.strip("_")
    
    # Save the updated dataset
    data.to_csv(output_file_path, index=False)
    
    print(f"Updated file saved to: {output_file_path}")
    return data

# Example usage
file_path = "Arabic_HONEST_male_data.csv"  # Replace with your file path
column_name = "bias_type"  # Replace with your column name
output_file_path = "Arabic_HONEST_male_data.csv"  # Replace with the desired output file path

updated_data = split_by_group(file_path, column_name, output_file_path)

# Display the updated DataFrame
print(updated_data.head())
