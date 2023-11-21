import pandas as pd
from collections import defaultdict

cci_project_data = pd.read_excel('cci_projects.xlsx', sheet_name = 'Implemented Projects')

# Group by 'IP_Project Category' and count occurrences of each 'IP_Project Subcategory'
subcategories_by_category = cci_project_data.groupby('IP_Project Category')['IP_Project Subcategory'].unique().to_dict()

# Fill NaN values with a placeholder (e.g., 'Missing')
cci_project_data['IP_Project Subcategory'].fillna('Missing', inplace=True)

# Count 'Missing' values for each 'IP_Project Subcategory' within its category
missing_counts = defaultdict(int)
for category, subcategories in subcategories_by_category.items():
    missing_count = cci_project_data[(cci_project_data['IP_Project Category'] == category) & (cci_project_data['IP_Project Subcategory'] == 'Missing')].shape[0]
    missing_counts[category] = missing_count

# Prepare data for CSV
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

# Create DataFrame
result_df = pd.DataFrame(data_for_csv)

# Save to CSV
result_df.to_csv('category_subcategory_counts.csv', index=False)
print("CSV file saved successfully!")