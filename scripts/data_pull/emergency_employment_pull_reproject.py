import csv
import pandas as pd
import os
pip install xlrd

# I could not consistently scrape the urls from the site, so I manually downloaded the files and placed them into a folder and then processed from there
# The code works with a folder of the downloaded 'EEO Detailed Occupations' county data from: https://labormarketinfo.edd.ca.gov/geography/demoaa.html
def xls_to_csv(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".xls"):
            # Construct the full paths for input and output
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".csv"
            output_path = os.path.join(output_folder, output_filename)

            # Read the Excel file
            df = pd.read_excel(input_path)

            # Find the index of the row containing 'Occupation Code'
            occupation_code_index = df[df.apply(lambda row: 'Occupation Code' in str(row), axis=1)].index

            # Check if 'Occupation Code' is found in the DataFrame
            if not occupation_code_index.empty:
                # Remove rows until 'Occupation Code'
                df = df.iloc[occupation_code_index[0]:]

                # Move the row with 'Occupation Code' to be the new header
                df.columns = df.iloc[0]

                # Drop the old header row
                df = df.iloc[1:]

                # Reset index after removing rows
                df.reset_index(drop=True, inplace=True)

                # Isolate rows that only contain desired occupation strings
                strings_to_isolate = ['Police officers 3870', 
                                      'Firefighting and prevention workers 3740',
                                      'Registered nurses 3255', 
                                      'Emergency medical technicians and paramedics 3401']

                # Filter rows based on the first column values
                df = df[df.iloc[:, 0].astype(str).str.strip().isin(strings_to_isolate)]

            else:
                print(f"Warning: 'Occupation Code' not found in {filename}. No conversion performed.")
                continue  # Skip further processing for this file
            
            
            # Write to CSV
            df.to_csv(output_path, index=False)
            print(f"Converted {filename} to {output_filename}, removed rows until 'Occupation Code', set new header, and isolated rows with specified strings")
            
xls_to_csv('downloaded_files','cleaned_csv_files', )

def process_csv_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Check if 'Subject' column exists in the DataFrame outside the loop
            if 'Subject' in df.columns:
                for index, row in df.iterrows():
                    # Check if the current row contains 'Total, both sexes' in the 'Subject' column
                    if 'Total, both sexes' in str(row['Subject']):
                        # Rename the value in the 'Subject' column with 'Occupation Total'
                        df.at[index + 1, 'Subject'] = 'Occupation Total'

                # Rename the third column to 'Value'
                df.columns.values[2] = 'Value'

                # Get rid of all columns past the third
                # Eliminate rows within the 'Subject' column that have unwanted data
                df = df.iloc[:, :3]
                df = df[~df['Subject'].str.contains('Percent')]
                df = df[~df['Subject'].str.contains('Number')]
                df = df[~df['Subject'].str.contains('Male')]
                df = df[~df['Subject'].str.contains('Female')]
                df = df[~df['Subject'].str.contains('Total, both sexes')]

                # Save the modified DataFrame to a new CSV file in the output folder
                output_file_path = os.path.join(output_folder, f"processed_{filename}")
                df.to_csv(output_file_path, index=False)

                # Display the modified DataFrame (optional)
                #print(df)

process_csv_files('cleaned_csv_files', 'final_cleaned_data')

# Dictionary to attribute a county to its respective file name, as there are no columns
# in the original files indicating the county
filename_to_county = {
    'processed_alameDetOcc.csv': 'Alameda',
    'processed_alpinDetOcc.csv': 'Alpine',
    'processed_amadoDetOcc.csv': 'Amador',
    'processed_calavDetOcc.csv': 'Calaveras',
    'processed_colusDetOcc.csv': 'Colusa',
    'processed_contrDetOcc.csv': 'Contra Costa',
    'processed_delnoDetOcc.csv': 'Del Norte',
    'processed_eldorDetOcc.csv': 'El Dorado',
    'processed_fresnDetOcc.csv': 'Fresno',
    'processed_glennDetOcc.csv': 'Glenn',
    'processed_humboDetOcc.csv': 'Humboldt',
    'processed_imperDetOcc.csv': 'Imperial',
    'processed_inyoDetOcc.csv': 'Inyo',
    'processed_kernDetOcc.csv': 'Kern',
    'processed_kingsDetOcc.csv': 'Kings',
    'processed_laDetOcc.csv': 'Los Angeles',
    'processed_lakeDetOcc.csv': 'Lake',
    'processed_lassenDetOcc.csv': 'Lassen',
    'processed_maderDetOcc.csv': 'Madera',
    'processed_marinDetOcc.csv': 'Marin',
    'processed_maripDetOcc.csv': 'Mariposa',
    'processed_mendoDetOcc.csv': 'Mendocino',
    'processed_merceDetOcc.csv': 'Merced',
    'processed_modocDetOcc.csv': 'Modoc',
    'processed_monoDetOcc.csv': 'Mono',
    'processed_monteDetOcc.csv': 'Monterey',
    'processed_napaDetOcc.csv': 'Napa',
    'processed_nevadDetOcc.csv': 'Nevada',
    'processed_oranDetOcc.csv': 'Orange',
    'processed_placeDetOcc.csv': 'Placer',
    'processed_plumaDetOcc.csv': 'Plumas',
    'processed_riveDetOcc.csv': 'Riverside',
    'processed_sacDetOcc.csv': 'Sacramento',
    'processed_sanbeDetOcc.csv': 'San Benito',
    'processed_sanbrDetOcc.csv': 'San Bernardino',
    'processed_sandiDetOcc.csv': 'San Diego',
    'processed_sanfrDetOcc.csv': 'San Francisco',
    'processed_sanjoDetOcc.csv': 'San Joaquin',
    'processed_sanluDetOcc.csv': 'San Luis Obispo',
    'processed_sanmaDetOcc.csv': 'San Mateo',
    'processed_santbDetOcc.csv': 'Santa Barbara',
    'processed_santcDetOcc.csv': 'Santa Clara',
    'processed_scruzDetOcc.csv': 'Santa Cruz',
    'processed_shastDetOcc.csv': 'Shasta',
    'processed_sierrDetOcc.csv': 'Sierra',
    'processed_siskiDetOcc.csv': 'Siskiyou',
    'processed_solanDetOcc.csv': 'Solano',
    'processed_sonomDetOcc.csv': 'Sonoma',
    'processed_staniDetOcc.csv': 'Stanislaus',
    'processed_sutteDetOcc.csv': 'Sutter',
    'processed_tehamDetOcc.csv': 'Tehama',
    'processed_triniDetOcc.csv': 'Trinity',
    'processed_tularDetOcc.csv': 'Tulare',
    'processed_tuoluDetOcc.csv': 'Tuolumne',
    'processed_ventuDetOcc.csv': 'Ventura',
    'processed_yoloDetOcc.csv': 'Yolo',
    'processed_yubaDetOcc.csv': 'Yuba'
                    }

def merge_data_single_csv(input_file, output_file): 
    # Create an empty DataFrame to store combined data
    combined_data = pd.DataFrame()

    # Loop through each file in the input folder
    for filename in os.listdir(input_file):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_file, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            county_value = None
            # Check if the entire filename matches any key in the dictionary
            if filename in filename_to_county:
                county_value = filename_to_county[filename]

            if county_value is not None:
                # Add a new column 'County' based on the dictionary value
                df['County'] = county_value

                # Add a new column 'SourceFile' to indicate the source file
                # df['SourceFile'] = filename

                # Append the DataFrame to the combined_data DataFrame
                #combined_data = combined_data.append(df, ignore_index=True)
                combined_data = pd.concat([combined_data, df], ignore_index=True)
    # Save the combined DataFrame to a new CSV file
    combined_data.to_csv(output_file, index=False)
    
merge_data_single_csv('final_cleaned_data', 'ca_emergency_employment.csv')
