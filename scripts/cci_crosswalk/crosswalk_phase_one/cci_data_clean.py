import pandas as pd
from collections import defaultdict

print('Reading in large excel file for data from May 2021 (first batch)')
print('')
cci_project_data = pd.read_excel('cci_projects.xlsx', sheet_name = 'Implemented Projects')

# Group by 'IP_Project Category' and count occurrences of each 'IP_Project Subcategory'
subcategories_by_category = cci_project_data.groupby('IP_Project Category')['IP_Project Subcategory'].unique().to_dict()
print('Isolated project category and unique subcategories within each category')
print('')
print('Also tracking empty rows with: Missing, which will be counted as well')
# Fill NaN values with a placeholder (e.g., 'Missing')
cci_project_data['IP_Project Subcategory'].fillna('Missing', inplace=True)

# Count 'Missing' values for each 'IP_Project Subcategory' within its category
missing_counts = defaultdict(int)
for category, subcategories in subcategories_by_category.items():
    missing_count = cci_project_data[(cci_project_data['IP_Project Category'] == category) & (cci_project_data['IP_Project Subcategory'] == 'Missing')].shape[0]
    missing_counts[category] = missing_count

# Prepare data for CSV
print('Counting how many times each category is entered and the number of unique subcategories within')
print('This could take some time')
print('')
data_for_csv = []
for category, subcategories in subcategories_by_category.items():
    category_count = len(cci_project_data[cci_project_data['IP_Project Category'] == category])
    for subcategory in subcategories:
        subcategory_count = cci_project_data[(cci_project_data['IP_Project Category'] == category) & (cci_project_data['IP_Project Subcategory'] == subcategory)].shape[0]
        data_for_csv.append({
            'Category': category,
            'Category_Count': category_count,
            'Subcategory': subcategory,
            'Subcategory_Count': subcategory_count,
            'Missing_Count': missing_counts[category]
        })
print('Interations complete, new csv will be made with desired columns can counts for each, including missing entires.')

# Create DataFrame
result_df = pd.DataFrame(data_for_csv)

# Save to CSV
result_df.to_csv('category_subcategory_counts_may_2021.csv', index=False)
print("CSV file saved successfully!")