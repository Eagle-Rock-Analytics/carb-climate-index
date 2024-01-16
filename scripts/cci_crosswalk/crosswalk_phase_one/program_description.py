import pandas as pd

print('Pulling large dataset, this could take a few minutes...')
cci_project_data = pd.read_excel('CCI_Projects_v05312021.xlsx', sheet_name = 'Implemented Projects')

print('')
print('pull complete, now isolating program descriptions')
print('')

# Counting unique entries in 'IP_Project Category'
unique_categories = cci_project_data['IP_Project Category'].nunique()

# Creating a new dictionary to store unique project descriptions for each subcategory or category
description_data = {}

# Looping through each unique 'IP_Project Category'
for category in cci_project_data['IP_Project Category'].unique():
    # Creating an empty dictionary to store descriptions for each subcategory or category
    description_counts = {}

    # Checking if there are subcategories for the current category
    subcategories = cci_project_data[cci_project_data['IP_Project Category'] == category]['IP_Project Subcategory'].unique()
    
    # Determining whether to fetch descriptions based on subcategories or category
    if len(subcategories) > 0:
        for subcategory in subcategories:
            # Collecting unique descriptions for each subcategory
            unique_desc = cci_project_data[(cci_project_data['IP_Project Category'] == category) & 
                                           (cci_project_data['IP_Project Subcategory'] == subcategory)]['ProgramDescription'].unique()
            # Storing descriptions in the dictionary
            description_counts[subcategory] = unique_desc
    else:
        # If there are no subcategories, collect descriptions for the category
        unique_desc = cci_project_data[cci_project_data['IP_Project Category'] == category]['ProgramDescription'].unique()
        # Storing descriptions in the dictionary
        description_counts[category] = unique_desc

    # Storing the descriptions for each subcategory or category
    description_data[category] = description_counts

# Writing the results to a new text file
output_file = 'program_descriptions.txt'

with open(output_file, 'w') as file:
    file.write(f"Number of unique 'IP_Project Category': {unique_categories}\n")
    file.write("Category Descriptions:\n")
    for category, desc in description_data.items():
        file.write(f"- Category: {category}\n")
        if isinstance(desc, dict):
            file.write("  - Subcategory Descriptions:\n")
            for subcategory, descriptions in desc.items():
                file.write(f"    - Subcategory: {subcategory}\n")
                file.write("      - Descriptions:\n")
                for description in descriptions:
                    file.write(f"        - {description}\n")
        else:
            file.write("  - Descriptions:\n")
            for description in desc:
                file.write(f"    - {description}\n")

print(f"Results saved to {output_file}")