import pandas as pd

print('Pulling large dataset, this could take a few minutes...')
cci_project_data = pd.read_excel('CCI_Projects_v05312021.xlsx', sheet_name = 'Implemented Projects')

print('')
print('pull complete, now filtering through desired columns to summarize dataset')
print('')

# Counting unique entries in 'IP_Project Category'
unique_categories = cci_project_data['IP_Project Category'].nunique()

total_unique_project_desc = cci_project_data['ProjectDescription'].nunique()

# Creating an empty dictionary to store counts for each unique 'IP_Project Category'
category_counts = {}

# Counting occurrences of each unique 'IP_Project Category' in the DataFrame
category_occurrences = cci_project_data['IP_Project Category'].value_counts()

# Looping through each unique 'IP_Project Category'
for category in cci_project_data['IP_Project Category'].unique():
    # Counting unique 'ProgramDescription' for each 'IP_Project Category'
    unique_program_desc = cci_project_data[cci_project_data['IP_Project Category'] == category]['ProgramDescription'].nunique()
    # Counting unique 'ProjectDescription' for each 'IP_Project Category'
    unique_project_desc = cci_project_data[cci_project_data['IP_Project Category'] == category]['ProjectDescription'].nunique()
    # Storing the counts in the dictionary
    category_counts[category] = {'Unique ProgramDescription': unique_program_desc, 'Unique ProjectDescription': unique_project_desc}
    
    # Creating an empty dictionary to store counts for each unique 'IP_Project Subcategory' within the category
    subcategory_counts = {}
    
    # Looping through each unique 'IP_Project Subcategory' within the 'IP_Project Category'
    for subcategory in cci_project_data[cci_project_data['IP_Project Category'] == category]['IP_Project Subcategory'].unique():
        # Counting occurrences of each unique 'IP_Project Subcategory' within the category
        subcategory_count = cci_project_data[(cci_project_data['IP_Project Category'] == category) & (cci_project_data['IP_Project Subcategory'] == subcategory)].shape[0]
        # Storing the count in the subcategory dictionary
        subcategory_counts[subcategory] = subcategory_count
    
    # Storing the subcategory counts for each category
    category_counts[category]['Subcategory Counts'] = subcategory_counts

# Writing the results to a text file
output_file = 'cci_project_summary.txt'

# Assuming category_occurrences is a dictionary containing your data
sorted_categories = sorted(category_occurrences.items(), key=lambda x: x[1], reverse=True)

# To just give IP Project occurances and number of each
'''
with open("your_file.txt", "w") as file:
    for category, count in sorted_categories:
        file.write(f"  - Number of IP Project '{category}' in dataset: {count}\n")
'''
# To give comprehensive summary stats
with open(output_file, 'w') as file:
    file.write(f"Number of unique 'IP_Project Category': {unique_categories}\n")
    file.write(f"Total number of unique 'ProjectDescription': {total_unique_project_desc}\n")
    file.write("Category Counts:\n")
    for category, counts in category_counts.items():
        file.write(f"- Category: {category}\n")
        file.write(f"  - Number of IP Project '{category}' in dataset: {category_occurrences[category]}\n")
        file.write(f"  - Number of unique Program descriptions: {counts['Unique ProgramDescription']}\n")
        file.write(f"  - Number of unique Project descriptions: {counts['Unique ProjectDescription']}\n")
        file.write("  - Subcategory Counts:\n")
        for subcategory, count in counts['Subcategory Counts'].items():
            file.write(f"    - {subcategory}: {count}\n")

print(f"Results saved to {output_file}")