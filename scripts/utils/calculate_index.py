import pandas as pd
import os
import sys
import glob
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# metric to indicator dictionaries
def indicator_dicts(domain):
    metric_to_indicator_society_dict = {
        'vulnerable_populations' : ['asthma', 
                                    'cardiovascular_disease', 
                                    'birth_weight',
                                    'education',
                                    'linguistic',
                                    'poverty', 
                                    'unemployment',
                                    'housing_burden',
                                    'imp_water_bodies',
                                    'homeless',
                                    'health_insurance',
                                    'ambulatory_disabilities',
                                    'cognitive_disabilities',
                                    'air conditioning',
                                    'Violent Crimes',
                                    'working outdoors', 
                                    '1miurban_10mirural',
                                    'american_indian',
                                    'over_65',
                                    'under_5',
                                    'household_financial_assistance'],

                'social_services' : ['blood',
                                    'hospitals',
                                    'care store',
                                    'engineering',
                                    'specialty trade',
                                    'repair',
                                    'mental_shortage',
                                    'primary_care',
                                    'narcotic'],

                'economic_health' : ['gini',
                                    'median_income',
                                    'hachman'] 
    }

    if domain == 'society':
        return metric_to_indicator_society_dict
    elif domain == 'natural':
        return metric_to_indicator_natural_dict
    elif domain == 'built':
        return metric_to_indicator_built_dict
    elif domain == 'governance':
        return metric_to_indicator_governance_dict
    elif domain == 'climate':
        return metric_to_indicator_climate_dict




def process_domain_csv_files(prefix, output_folder, meta_csv, merged_output_file):
    '''
    Pulls metric csv files based on domain prefix variable and merges all together based on shared GEOID column. NaN values within the GEOID column are removed and infinite values (if any) in other columns are adjusted to NaN values. Lastly, an uninhabited island tract is also given NaN metric values.

    Parameters
    ----------
    prefix: str
        Shared prefix for the desired domain csv files to call in:
            'society_'
            'built_'
            'governance_'
            'climate_'  
    output_folder: str
        name of the folder to store pulled domain csv files
    meta_csv: str
        local path to the metadata pipeline
    merged_output_file: str
        desired name of merged output csv file
    '''

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the metadata CSV
    df = pd.read_csv(meta_csv)

    # Get the list of metric file names and corresponding 'High value result' entries
    metric_files = df[['Metric file name', 'High value result (vulnerable or resilient)']]

    # Find all CSV files starting with the provided prefix and matching the metric file names
    source_files = [file for file in glob.glob(f'{prefix}*.csv') if os.path.basename(file) in metric_files['Metric file name'].values]

    # Iterate through the source files and process them
    for file in source_files:
        # Get the 'High value result (vulnerable or resilient)' entry for the current file
        column_result = metric_files.loc[metric_files['Metric file name'] == os.path.basename(file), 'High value result (vulnerable or resilient)'].values[0]

        # Load the CSV file
        csv_df = pd.read_csv(file)

        # Get the last column name
        last_column = csv_df.columns[-1]

        # Append the column result to the last column name
        csv_df.rename(columns={last_column: f"{last_column}_{column_result}"}, inplace=True)

        # Construct the destination file path
        destination_path = os.path.join(output_folder, os.path.basename(file))

        # Save the modified CSV to the output folder
        csv_df.to_csv(destination_path, index=False)

        # Remove the original file
        os.remove(file)

    print(f"Processed and saved {len(source_files)} CSV files.")

    # Delete all CSV files in the current directory that are not in the output folder
    current_files = glob.glob('*.csv')
    for file in current_files:
        if file not in [os.path.basename(f) for f in source_files]:
            os.remove(file)

    print(f"Deleted {len(current_files) - len(source_files)} local non-relevant CSV files.")
    print('')

    # --- Additional Processing: Merging CSV Files ---

    # Get a list of all CSV files in the output folder
    csv_files = glob.glob(os.path.join(output_folder, '*.csv'))

    # Initialize an empty DataFrame for merging
    merged_df = pd.DataFrame()

    # Iterate through each CSV file and merge them on the 'GEOID' column
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Rename 'GEO_ID', 'tract', 'TRACT', 'Census_Tract', 'GEOID', 'USCB_GEOID' to 'GEOID' if they exist
        rename_cols = ['GEO_ID', 'tract', 'TRACT', 'Census_Tract', 'census_tract', 'USCB_GEOID']
        for col in rename_cols:
            if col in df.columns:
                df.rename(columns={col: 'GEOID'}, inplace=True)
                break
        
        # Keep only the 'GEOID' and the last column from each file
        last_column = df.columns[-1]
        df = df[['GEOID', last_column]]
        
        # Merge the DataFrame with the existing merged DataFrame
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='GEOID', how='outer')

    # Drop rows where 'GEOID' is NaN
    merged_df = merged_df.dropna(subset=['GEOID'])

    # Convert census tract to string and eliminate scientific notation default
    merged_df['GEOID'] = merged_df['GEOID'].dropna().apply(lambda x: '{:.0f}'.format(x))

    # Convert all values within the island tract (near San Francisco) to NaN, as it is uninhabited 
    island_tract = '6075980401'
    merged_df.loc[merged_df['GEOID'] == island_tract, merged_df.columns != 'GEOID'] = np.nan

    # Check if all entries within the island tract are NaN
    island_row = merged_df.loc[merged_df['GEOID'] == island_tract]
    if island_row.iloc[:, 1:].isnull().all().all():
        print(f"All entries within the island tract ({island_tract}) are NaN.")
    else:
        print(f"Some entries within the island tract ({island_tract}) are not NaN.")

    merged_df['GEOID'] = merged_df['GEOID'].apply(lambda x: '0' + str(x))
    merged_df['GEOID'] = merged_df['GEOID'].astype(str).apply(lambda x: x.rstrip('0').rstrip('.') if '.' in x else x)
    # Selecting only numeric columns
    numeric_df = merged_df.select_dtypes(include=[np.number])

    # Counting infinite values
    num_infinite = np.isinf(numeric_df).sum().sum()

    print(f"\nNumber of infinite entries in the DataFrame: {num_infinite}")
    print('Replacing infinite entries (if any) with NaN')

    # Replace infinite values with NaN
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Selecting only numeric columns
    numeric_df = merged_df.select_dtypes(include=[np.number])

    # Counting infinite values
    num_infinite = np.isinf(numeric_df).sum().sum()
    print(f"Number of infinite entries in the DataFrame: {num_infinite}")

    print(f"\nFile processing complete, dataframe will now be saved as a .csv")
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(merged_output_file, index=False)

    print(f"Processed CSV saved as {merged_output_file}")

def weight_domains(df, society, built, natural):
    '''
    Calculates the weighting scheme, based on input parameters:
    society, built, and natural
    '''
    governance_col = 'DUMMY_governance_summed_indicators_min_max_standardized'
    society_adjusted_col = 'DUMMY_society_tract_adjusted'
    built_adjusted_col = 'DUMMY_built_tract_adjusted'
    natural_adjusted_col = 'DUMMY_natural_tract_adjusted' 

    weighting = (
        df[governance_col] + 
        (society * (df[society_adjusted_col] * df[governance_col])) +
        (built * (df[built_adjusted_col] * df[governance_col])) +
        (natural * (df[natural_adjusted_col] * df[governance_col]))
    )

    df['calcrai_weighted'] = weighting
    return df


def calculate_index(df):
    '''Calcutes the Cal-CRAI index'''
    
    df['calcrai_score'] = df['calcrai_weighted'] / df['acute_risk']

    # testing for 0 values --> divide error
    df.loc[df['acute_risk'] == 0, 'calcrai_score'] = 0
    
    return df


def format_df(df):
    '''
    Minor clean-up of pandas df -- can be resolved in future version
    Demo purposes only, at present
    '''
    if "field_1" in df.columns:
        df = df.drop(columns='field_1') # drops extra field
        
    df['GEOID'] = '0' + df['GEOID'] # formats GEOID column to match shapefile (has an extra 0 in front)

    for i in df.columns:
        exclude = ["geometry", "GEOID"]
        if i not in exclude:
            df[i] = df[i].astype(float) # changes type of core columns to float from string

    return df

def handle_outliers(df, domain_prefix, summary_stats=True, print_all_vals=False):
    # Columns to process (exclude 'GEOID')
    columns_to_process = [col for col in df.columns if col != 'GEOID']
    
    # Dictionary to store counts of adjusted rows
    adjusted_counts = {}

    for column in columns_to_process:
        # Convert the column to numeric, forcing any errors to NaN
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            print(f"Column '{column}' has no IQR (Q1 == Q3). Skipping outlier handling for this column.")
            continue

        max_fence = Q3 * 3
        min_fence = Q1 * -3

        if summary_stats:
            print(f'For column {column}:')
            print(f'  Q1 (25th percentile): {Q1}')
            print(f'  Q3 (75th percentile): {Q3}')
            print(f'  IQR: {IQR}')
            print(f'  Max fence: {max_fence}')
            print(f'  Min fence: {min_fence}')

        # Identify outliers
        outliers = df[(df[column] > max_fence) | (df[column] < min_fence)]
        
        # Print outliers and their corresponding 'GEOID'
        if not outliers.empty:
            if print_all_vals: # if all values printed to screen is desired
                print(f"Outliers detected in column '{column}':")
                for _, row in outliers.iterrows():
                    print(f"   GEOID: {row['GEOID']}, value: {row[column]}")

        # Count the number of adjustments
        count_adjusted = df[(df[column] > max_fence) | (df[column] < min_fence)].shape[0]
        adjusted_counts[column] = count_adjusted
        
        # Clip the outliers
        df[column] = df[column].clip(lower=min_fence, upper=max_fence)
    
    # Save the updated DataFrame back to CSV
    # close out
    handle_outlier_csv = "no_outlier_{}metrics.csv".format(domain_prefix)
    print(f"Processed and saved {handle_outlier_csv} with outlier handling.")
    df.to_csv(handle_outlier_csv, index=False)
    
    # Print the adjusted counts
    if summary_stats:
        print("Number of rows adjusted per column:")
        for column, count in adjusted_counts.items():
            print(f"  {column}: {count}")

    return df

def min_max_standardize(df, cols_to_run_on, tolerance=1e-9):
    '''
    Calculates min and max values for specified columns, then calculates
    min-max standardized values with a tolerance for floating-point precision errors.

    Parameters
    ----------
    df: DataFrame
        Input dataframe   
    cols_to_run_on: list
        List of columns to calculate min, max, and standardize
    tolerance: float
        Tolerance value for checking if standardized values are within the [0, 1] range
    '''
    all_good = True  # Flag to track if all columns are within range

    for col in cols_to_run_on:
        max_value = df[col].max()
        min_value = df[col].min()
        
        # Get min-max values, standardize, and add columns to df
        prefix = col  # Using the column name as the prefix for new columns
        df[f'{prefix}_min'] = min_value
        df[f'{prefix}_max'] = max_value
        df[f'{prefix}_min_max_standardized'] = ((df[col] - min_value) / (max_value - min_value))

        # Check if the new standardized column values are between 0 and 1, ignoring NaN values
        standardized_col = df[f'{prefix}_min_max_standardized']
        is_within_range = standardized_col.dropna().between(-tolerance, 1 + tolerance)

        if not is_within_range.all():
            all_good = False
            out_of_bounds = standardized_col[~is_within_range]
            print(f"Warning: Column '{prefix}_min_max_standardized' has values outside the [0, 1] range (considering tolerance).")
            print(out_of_bounds)

        # Drop the original column
        df.drop(columns=[col], inplace=True)
    
    # Print a summary at the end
    if all_good:
        print("All standardized columns are within the [0, 1] range (considering tolerance).")
    else:
        print("Some columns have values outside the [0, 1] range.")

    return df

def compute_averaged_indicators(df, metric_to_indicator_dict):
    '''
    Computes the average of selected columns based on keywords for each indicator in the dictionary
    and stores the result in a new DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with standardized metrics.
    metric_to_indicator_dict : dict
        Dictionary where keys are indicator names and values are lists of keywords to match column names.
    
    Returns
    -------
    DataFrame
        DataFrame with averaged indicators and 'GEOID' column.
    '''
    
    # Create an empty DataFrame to store the results
    avg_indicator_metrics = pd.DataFrame()

    # Iterate through the items of the dictionary
    for indicator, keywords in metric_to_indicator_dict.items():
        # Filter columns based on the keyword values for the current indicator
        indicator_columns = [col for col in df.columns if any(keyword in col for keyword in keywords)]
        
        # Compute the average of the selected columns
        averaged_values = df[indicator_columns].mean(axis=1)
        
        # Store the averaged values in the result DataFrame with the indicator name as the column name
        avg_indicator_metrics[indicator] = averaged_values

    # Include the 'GEOID' column from the original DataFrame
    avg_indicator_metrics['GEOID'] = df['GEOID']
    
    # Reorder the columns to have 'GEOID' as the first column
    avg_indicator_metrics = avg_indicator_metrics[['GEOID'] + [col for col in avg_indicator_metrics.columns if col != 'GEOID']]
    # print(avg_indicator_metrics)
   
    return avg_indicator_metrics

def compute_summed_indicators(df, columns_to_sum):
    '''
    Computes the sum of the specified columns in the input DataFrame and stores the result in a new DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with indicators.
    columns_to_sum : list
        List of column names to sum.
    
    Returns
    -------
    DataFrame
        DataFrame with summed indicators and 'GEOID' column.
    '''
    
    # Ensure columns_to_sum is a list of column names
    if not isinstance(columns_to_sum, list):
        raise TypeError("columns_to_sum must be a list of column names.")
    
    # Create a new DataFrame to store the results
    summed_indicators_df = pd.DataFrame()

    # Calculate the sum of the specified columns
    summed_values = df[columns_to_sum].sum(axis=1)

    # Store the summed values in the result DataFrame with the specified column name
    summed_indicators_df['summed_indicators_society_economy_domain'] = summed_values

    # Include the 'GEOID' column from the original DataFrame
    summed_indicators_df['GEOID'] = df['GEOID']

    # Reorder the columns to have 'GEOID' as the first column
    summed_indicators_df = summed_indicators_df[['GEOID', 'summed_indicators_society_economy_domain']]

    # Print the resulting DataFrame (optional)
    # print(summed_indicators_df)
    print('Indicator sum min value:', summed_indicators_df['summed_indicators_society_economy_domain'].min())
    print('Indicator sum max value:', summed_indicators_df['summed_indicators_society_economy_domain'].max())

    return summed_indicators_df


def add_census_tracts(df):
    '''merges the census tract boundaries to the processed dataframe'''

    # read in census tracts
    census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/"
    ca_boundaries = gpd.read_file(census_shp_dir)
    ca_boundaries['GEOID'] = ca_boundaries['GEOID'].astype(str)

    # merge to df
    df_merge = df.merge(ca_boundaries, on='GEOID')

    # conver to correct CRS
    gdf = gpd.GeoDataFrame(df_merge, geometry='geometry', crs=4269)

    return gdf


def domain_summary_stats(gdf, domain):
    # locate the min-max standardized column
    if domain == 'society_':
        domain = 'society_economy'
    col = f'summed_indicators_{domain}_domain_min_max_standardized'

    # summary stats
    print(f'Median {domain} domain value: {gdf[col].median()}')
    print(f'Mean {domain} domain value: {gdf[col].mean()}')
