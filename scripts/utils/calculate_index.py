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

def handle_outliers(df, output_csv):
    # Columns to process (exclude 'census_tract')
    columns_to_process = [col for col in df.columns if col != 'census_tract']
    
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

        print(f'For column {column}:')
        print(f'  Q1 (25th percentile): {Q1}')
        print(f'  Q3 (75th percentile): {Q3}')
        print(f'  IQR: {IQR}')
        print(f'  Max fence: {max_fence}')
        print(f'  Min fence: {min_fence}')

        # Identify outliers
        outliers = df[(df[column] > max_fence) | (df[column] < min_fence)]
        
        # Print outliers and their corresponding 'census_tract'
        if not outliers.empty:
            print(f"Outliers detected in column '{column}':")
            for _, row in outliers.iterrows():
                print(f"census_tract: {row['census_tract']}, value: {row[column]}")

        # Count the number of adjustments
        count_adjusted = df[(df[column] > max_fence) | (df[column] < min_fence)].shape[0]
        adjusted_counts[column] = count_adjusted
        
        # Clip the outliers
        df[column] = df[column].clip(lower=min_fence, upper=max_fence)
    
    # Save the updated DataFrame back to CSV
    df.to_csv(output_csv, index=False)
    
    # Print the adjusted counts
    print("Number of rows adjusted per column:")
    for column, count in adjusted_counts.items():
        print(f"  {column}: {count}")

    return df

def min_max_standardize(df, cols_to_run_on):
    '''
    Calculates min and max values for specified columns, then calculates
    min-max standardized values.

    Parameters
    ----------
    df: DataFrame
        Input dataframe   
    cols_to_run_on: list
        List of columns to calculate min, max, and standardize
    '''
    for col in cols_to_run_on:
        max_value = df[col].max()
        min_value = df[col].min()

        # Get min-max values, standardize, and add columns to df
        prefix = col # Extracting the prefix from the column name
        df[f'{prefix}_min'] = min_value
        df[f'{prefix}_max'] = max_value
        df[f'{prefix}_min_max_standardized'] = ((df[col] - min_value) / (max_value - min_value))
        
        # note to add checker to make sure new min_max column values arent < 0 >
        
        
        # Drop the original columns
        df.drop(columns=[col], inplace=True)
     
    return df