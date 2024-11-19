import pandas as pd
import os
import sys
import glob
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

def indicator_dicts(domain):
    '''
    Contains domain specific dictionaries that attribute a metric to its indicator.
    
    Parameters
    ----------
    domain: str
        calls a specific metric to indicator dictionary for one of the five possible domains:
            - 'society_economy'
            - 'natural'
            - 'built'
            - 'governance'
            - 'climate'
    '''
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
                                    'hachman']}
    metric_to_indicator_built_dict = {
        'communication' :           ['low_internet',
                                    'cellular_towers',
                                    'microwave_towers',
                                    'mobile_towers',
                                    'paging_towers',
                                    'radio_towers',
                                    'tv_contour'],

        'housing_vacancy_quality' : ['housing_before_1980',
                                    'mobile_homes',
                                    'kitchen_facilities',
                                    'vacant_housing'],

        'transportation' :          ['airports',
                                    'bottlenecks',
                                    'bridges',
                                    'highway',
                                    'railway'],

        'utilities' :               ['power_plant',
                                    'psps_event',
                                    'underground_transmission',
                                    'wastewater_facilities']}
    metric_to_indicator_natural_dict = {
            'natural_resource_conservation' : ['protected_areas'],

            'agricultural_productivity_conservation' : ['ssma',
                                                        'esi_mean'],
                                                        
            'ecosystem_condition' :               ['air_quality',
                                                'SpBioWtEco',
                                                'impervious',
                                                'vulnerable_soils',
                                                'vulnerable_drought',
                                                'vulnerable_fire']}
    metric_to_indicator_governance_dict = {
            'emergency_response' :  ['fire_stations',
                                    'medical_technicians',
                                    'firefighting',
                                    'police_officers',
                                    'registered_nurses'
                                    ],

            'personal_preparedness' : ['flood_policies',
                                        'mortgage',
                                        'prepared_for_general_disaster',
                                        'prepared_without_power',
                                        'prepared_without_water'
                                    ],

            'community_preparedness' :  ['fuel_reduction',
                                    'nfip_participation',
                                    'hazard_mitigation',
                                    'fuel_reduction'
                                    ],

            'natural_resource_conservation' :  ['timber_management'
                                                'sampled_wells'
                                            ]}
    metric_to_indicator_climate_dict = {
                    "exposure" :   ['drought_coverage_percentage',
                                    'change_in_drought_years',
                                    'percent_weeks_drought',
                                    'precip_99percentile',
                                    'surface_runoff',
                                    'floodplain_percentage',
                                    'median_flood_warning_days',
                                    'mean_change_annual_heat_days',
                                    'mean_change_annual_warm_nights',
                                    'median_heat_warning_days',
                                    'slr_vulnerability_delta_percentage_change',
                                    'slr_fire_stations_count_metric',
                                    'slr_police_stations_count_metric',
                                    'slr_schools_count_metric',
                                    'slr_hospitals_count_metric',
                                    'slr_vulnerable_wastewater_treatment_count',
                                    'building_exposed_slr_count',
                                    'slr_vulnerable_building_content_cost',
                                    'change_ffwi_days',
                                    'median_red_flag_warning_days'
                    ],
                    "loss"  :  ['drought_crop_loss_acres',
                                'drought_crop_loss_indemnity_amount',
                                'avg_flood_insurance_payout_per_claim',
                                'estimated_flood_crop_loss_cost',
                                'total_flood_fatalities',
                                'mean_change_cold_days',
                                'heat_crop_loss_acres',
                                'heat_crop_loss_indemnity_amount',
                                'avg_age_adjust_heat_hospitalizations_per_10000',
                                'rcp_4.5__50th_percent_change',
                                'burn_area_m2',
                                'average_damaged_destroyed_structures_wildfire',
                                'average_annual_fatalities_wildfire'
    ]}
   
    if domain == 'society_economy':
        return metric_to_indicator_society_dict
    elif domain == 'natural':
        return metric_to_indicator_natural_dict
    elif domain == 'built':
        return metric_to_indicator_built_dict
    elif domain == 'governance':
        return metric_to_indicator_governance_dict
    elif domain == 'climate':
        return metric_to_indicator_climate_dict

def process_domain_csv_files(prefix, input_folder, output_folder, meta_csv, merged_output_file):
    '''
    Pulls metric csv files based on domain prefix variable from the input folder, merges all together based on shared GEOID column. 
    NaN values within the GEOID column are removed, and infinite values (if any) in other columns are adjusted to NaN values.
    Lastly, an uninhabited island tract is also given NaN metric values.
    
    Parameters
    ----------
    prefix: str
        Shared prefix for the desired domain csv files to call in:
            'society_'
            'built_'
            'governance_'
            'climate_'
    input_folder: str
        Name of the folder that is storing all metric csv files
    output_folder: str
        Name of the folder to store pulled domain specific csv files.
    meta_csv: str
        Local path to the metadata pipeline.
    merged_output_file: str
        Desired name of merged output csv file.
    '''

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the metadata CSV
    df = pd.read_csv(meta_csv)

    # Get the list of metric file names and corresponding 'High value result' entries
    metric_files = df[['Metric file name', 'High value result (vulnerable or resilient)']]

    # Dictionary to hold metric column names categorized by 'vulnerable' and 'resilient'
    #global metric_vulnerable_resilient_dict
    metric_vulnerable_resilient_dict = {'vulnerable': [], 'resilient': []}


    # Find all CSV files starting with the provided prefix in the input folder and matching the metric file names
    source_files = [file for file in glob.glob(os.path.join(input_folder, f'{prefix}*.csv')) 
                    if os.path.basename(file) in metric_files['Metric file name'].values]

    # Iterate through the source files and process them
    for file in source_files:
        # Get the 'High value result (vulnerable or resilient)' entry for the current file
        column_result = metric_files.loc[metric_files['Metric file name'] == os.path.basename(file), 'High value result (vulnerable or resilient)'].values[0]

        # Load the CSV file
        csv_df = pd.read_csv(file)

        # Get the last column name
        last_column = csv_df.columns[-1]

        # Add the column name to the corresponding category in the dictionary
        metric_vulnerable_resilient_dict[column_result].append(last_column)

        # Construct the destination file path
        destination_path = os.path.join(output_folder, os.path.basename(file))

        # Save the modified CSV to the output folder
        csv_df.to_csv(destination_path, index=False)

        # Remove the original file
        os.remove(file)

    print(f"Processed and saved {len(source_files)} CSV files within {prefix}domain.")

    print('\nMetric resilience/vulnerable dictionary created and called: metric_vulnerable_resilient_dict')

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
        rename_cols = ['GEO_ID', 'GEOID', 'tract', 'TRACT', 'Census_Tract', 'census_tract', 'USCB_GEOID', 'Unnamed: 0']
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
        print('')
        print(f"All entries within the island tract ({island_tract}) are NaN.")
    else:
        print('')
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

    # Counting infinite values after replacement
    num_infinite = np.isinf(numeric_df).sum().sum()
    print(f"Number of infinite entries in the DataFrame after replacement: {num_infinite}")

    print(f"\nFile processing complete, dataframe will now be saved as a .csv")
    
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(merged_output_file, index=False)

    print(f"Processed CSV saved as {merged_output_file}")
    return metric_vulnerable_resilient_dict

def print_index_summary(df, column):
    '''
    Calculates the min, max, mean, and median Cal-CRAI values from a specified dataframe column.
    
    Parameters
    ----------
    df: DataFrame
        Input dataframe
    column: str
        Column in which summary stats are to be calculated on
    '''

    print('Min score / less resilience: ', df[column].min())
    print('Max score / more resilience: ', df[column].max())
    print('Mean score / average resilience: ', df[column].mean())
    print('Median score / median resilience: ', df[column].median())

def weight_domains(df, society, built, natural):
    '''
    Calculates the weighted numerator portion of the Cal-CRAI, based on numeric input parameters:
    society, built, and natural.
    
    Parameters
    ----------
    df: Dataframe
        Input dataframe
    society: int
        Weighting modifier for society domain
    built: int
        Weighting modifier for built domain
    natural: int
        Weighting modifier for natural domain
    '''
    governance_col = 'governance_domain_index'
    society_adjusted_col = 'society_economy_tract_adjusted'
    built_adjusted_col = 'built_tract_adjusted'
    natural_adjusted_col = 'natural_systems_tract_adjusted' 

    weighting = (
        df[governance_col] + 
        (society * (df[society_adjusted_col] * df[governance_col])) +
        (built * (df[built_adjusted_col] * df[governance_col])) +
        (natural * (df[natural_adjusted_col] * df[governance_col]))
    )

    df['calcrai_weighted'] = weighting
    return df

def calculate_weighted_index(df, climate_column):
    '''
    Calcutes the weighted scenario(s) for the Cal-CRAI with 'calcrai_weighted' being the
    presumed numerator column name in the input dataframe.
    
    Parameters
    ----------
    df: DataFrame
        Input dataframe  
    climate_column: str
        Climate column residing within the input df, it is the denominator of the Cal-CRAI calculation
    '''
    # divide by climate domain
    df['calcrai_score'] = df['calcrai_weighted'] / df[climate_column]

    # testing for 0 values --> divide error
    df.loc[df[climate_column] == 0, 'calcrai_score'] = 0
    
    return df

def calculate_equal_weighted_index(df):
    '''
    Calculates the equally weighted scenario for the Cal-CRAI with each domain coefficient
    within the calculation being '1'.
    
    Parameters
    ----------
    df: DataFrame
        Input dataframe  
    '''
    governance_col = 'governance_domain_index'
    society_adjusted_col = 'society_economy_tract_adjusted'
    built_adjusted_col = 'built_tract_adjusted'
    natural_adjusted_col = 'natural_systems_tract_adjusted' 

    weighting = (
        df[governance_col] + 
        (1 * (df[society_adjusted_col] * df[governance_col])) +
        (1 * (df[built_adjusted_col] * df[governance_col])) +
        (1 * (df[natural_adjusted_col] * df[governance_col]))
    )
    df['calcrai_weighted'] = weighting

    # divide by climate domain
    df['calcrai_score'] = df['calcrai_weighted'] / df['climate_risk']

    # testing for 0 values --> divide error
    df.loc[df['climate_risk'] == 0, 'calcrai_score'] = 0
    
    return df

def format_df(df):
    '''
    Minor clean-up of pandas df -- can be resolved in future version
    Demo purposes only, at present

    Parameters
    ----------
    df: DataFrame
        Input dataframe  
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
    '''
    Identifies 25th and 75th percentiles, then calculates the interquartile range
    for all columns not named 'GEOID'. Upper and lower fences are created by multiplying
    the IQR by 3 and -3 respectively. Values within each column that exceed its fencing range
    are adjusted to that fence value.  
    
    Parameters
    ----------
    df: DataFrame
        Input dataframe
    domain_prefix: str
        domain prefix name, strictly to name the output csv file
    summary_stats: bool
        True/False boolean (default is True) that prints number of values that were fenced 
        for exceeding the bounds.
    print_all_vals: bool
        True/False boolean (default is False) that prints fencing values for each column. 
    '''

    # Convert all columns except 'GEOID' to numeric
    for column in df.columns:
        if column != 'GEOID':
            df[column] = pd.to_numeric(df[column], errors='coerce')

    # Columns to process (exclude 'GEOID' and those containing words to skip)
    columns_to_process = [
        col for col in df.columns 
        if col != 'GEOID'
    ]
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
    #df.to_csv(handle_outlier_csv, index=False)
    
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
        # Convert the column to numeric, forcing any errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

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

def compute_summed_climate_indicators(df, metric_to_indicator_dict):
    '''
    Computes the sum of columns grouped together based on keywords for 
    exposure or loss indicators within the climate domain using a dictionary,
    then stores the result in a new DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame with standardized metrics.
    metric_to_indicator_dict : dict
        Dictionary where keys are indicator names and values
        are lists of keywords to match column names.
    
    Returns
    -------
    DataFrame
        DataFrame with summed climate indicators and 'GEOID' column.
    '''
    
    # Create an empty DataFrame to store the results
    summed_indicator_metrics = pd.DataFrame()

    # Iterate through the items of the dictionary
    for indicator, keywords in metric_to_indicator_dict.items():
        # Filter columns based on the keyword values for the current indicator
        indicator_columns = [col for col in df.columns if any(keyword in col for keyword in keywords)]
        
        # Compute the sum of the selected columns
        summed_values = df[indicator_columns].sum(axis=1)
        
        # Store the summed values in the result DataFrame with the indicator name as the column name
        summed_indicator_metrics[indicator] = summed_values

    # Include the 'GEOID' column from the original DataFrame
    summed_indicator_metrics['GEOID'] = df['GEOID']
    
    # Reorder the columns to have 'GEOID' as the first column
    summed_indicator_metrics = summed_indicator_metrics[['GEOID'] + [col for col in summed_indicator_metrics.columns if col != 'GEOID']]
    # print(avg_indicator_metrics)
   
    return summed_indicator_metrics

def compute_summed_indicators(df, columns_to_sum, domain_prefix):
    '''
    Computes the sum of the specified columns in the input DataFrame and 
    stores the result in a new DataFrame.

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

    summed_indicator_name = f'summed_indicators_{domain_prefix}domain'

    # Store the summed values in the result DataFrame with the specified column name
    summed_indicators_df[summed_indicator_name] = summed_values

    # Include the 'GEOID' column from the original DataFrame
    summed_indicators_df['GEOID'] = df['GEOID']

    # Reorder the columns to have 'GEOID' as the first column
    summed_indicators_df = summed_indicators_df[['GEOID', summed_indicator_name]]

    # Print the resulting DataFrame (optional)
    # print(summed_indicators_df)
    print('Indicator sum min value:', summed_indicators_df[summed_indicator_name].min())
    print('Indicator sum max value:', summed_indicators_df[summed_indicator_name].max())

    return summed_indicators_df

def add_census_tracts(df):
    '''
    Merges the census tract boundaries to the processed dataframe.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame with standardized metrics.
    '''

    # read in census tracts
    census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/"
    ca_boundaries = gpd.read_file(census_shp_dir)
    ca_boundaries['GEOID'] = ca_boundaries['GEOID'].astype(str)

    # merge to df
    df_merge = df.merge(ca_boundaries, on='GEOID')

    # conver to correct CRS
    gdf = gpd.GeoDataFrame(df_merge, geometry='geometry', crs=4269)

    return gdf

def domain_summary_stats(gdf, column):
    '''
    Merges the census tract boundaries to the processed dataframe.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame following the merge withe census tract
        geodataframe.
    column : str
        Name of the gdf domain score column you want stats for.
    '''
    print(f'Median {column} domain value: {gdf[column].median()}')
    print(f'Mean {column} domain value: {gdf[column].mean()}')