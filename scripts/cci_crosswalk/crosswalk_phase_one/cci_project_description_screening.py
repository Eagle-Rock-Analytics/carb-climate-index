import pandas as pd

print('Pulling large dataset, this could take a few minutes...')
cci_project_data = pd.read_excel('CCI_Projects_v05312021.xlsx', sheet_name = 'Implemented Projects')

print('')
print('pull complete, now filtering through unique project descriptions and generating statistics')
print('')

climate_risk_dict = {
    'wildfire mitigation': ['wildfire', 'Wildfire', 'burn', 'controlled burns', 'firefighting', 'forest', 'reforest', 'reforestation', 'fire', 'vegetation management', 'roadside brushing', 'fuel break'],
    'sea level rise mitigation': ['sea level rise', 'slr', 'erosion', 'seawall', 'riparian', 'seawalls', 'shoreline', 'wetland', 'mangrove'],
    'extreme heat mitigation': ['heat', 'extreme heat', 'shade', 'shading', 'cooling center', 'cooling centers', 'heat-resistant', 'heat resistant', 'heat reducing', 'heat-reducing'],
    'drought mitigation': ['drought', 'dry', 'irrigation', 'soil moisture', 'rainwater harvest', 'rainwater harvesting', 'water storage', 'water allocation', 'water management'],
    'inland flooding mitigation': ['flood', 'flooding', 'inland flood', 'inland flooding', 'floodplain', 'floodproof', 'floodproofing', 'elevated flood', 'flood barrier', 'flood barriers', 'drainage'],
    'greenhouse gas mitigation': ['ghg', 'GHG', 'greenhouse gas', 'emission', 'emissions', 'carbon sequestration', 'electrification', 'carbon capture', 'solar power', 'wind energy', 'hydroelectricity', 'geothermal energy', 'biomass energy', 'Energy-efficiency']
}

# Function to check for keywords and return matched categories
def find_categories(description):
    flagged_categories = []
    for category, keywords in climate_risk_dict.items():
        for keyword in keywords:
            if keyword.lower() in description.lower():
                flagged_categories.append(category)
                break  # If a keyword is found, break to avoid duplicate category entry
    return ','.join(flagged_categories)

# Replace NaN values in 'ProjectDescription' column with 'missing'
cci_project_data['ProjectDescription'].fillna('missing', inplace=True)

# Extract unique project descriptions
unique_descriptions = cci_project_data['ProjectDescription'].unique()

# Create an empty DataFrame to accumulate results
results_df = pd.DataFrame(columns=['IP_Project Category', 'IP_Project Subcategory', 'ProjectDescription', 'Flagged keywords'])

# Filter cci_project_data based on unique_descriptions
filtered_data = cci_project_data[cci_project_data['ProjectDescription'].isin(unique_descriptions)]

total_rows_processed = len(unique_descriptions)
total_rows_flagged = 0
entries_counter = {category: 0 for category in climate_risk_dict}
multiple_keys_count = 0
other_count = 0

for description in unique_descriptions:
    flagged_categories = find_categories(description)
    
    if flagged_categories:
        categories_list = flagged_categories.split(',')
        total_rows_flagged += 1
        
        if len(categories_list) > 1:
            multiple_keys_count += 1
        
        for category in categories_list:
            entries_counter[category] += 1
            
        # Find the rows that match the current description and extract other relevant columns
        matched_rows = filtered_data[filtered_data['ProjectDescription'] == description][['ProjectDescription', 'IP_Project Category', 'IP_Project Subcategory']]
        
        # Add all flagged categories to the matched rows
        matched_rows['Flagged keywords'] = ','.join(categories_list)
        
        # Append the matched rows to the results DataFrame
        results_df = pd.concat([results_df, matched_rows], ignore_index=True)
    else:
        other_count += 1

# Exporting data to a new CSV file
results_df.drop_duplicates(inplace=True)  # Remove duplicates
results_df.to_csv('flagged_keywords_results.csv', index=False)

# Print counts
print(f"Total Rows Processed (unique descriptions): {total_rows_processed}")
print(f"Total Rows Flagged (caught from dictionary): {total_rows_flagged}")
for category, count in entries_counter.items():
    print(f"{category}: {count}")
print(f"Other: {other_count}")
print(f"Multiple Keys Found: {multiple_keys_count}")