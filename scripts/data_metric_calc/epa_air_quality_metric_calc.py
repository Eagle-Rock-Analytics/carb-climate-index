import pandas as pd
import os
import csv

def process_air_quality_data(input_folder, filtered_csv_file, final_output_name):
    '''
    Unhealthy air quality days metric calculation. 'Total Unhealthy AQI days' are to sum of days in which the AQI value for a given
    county is >100 AQI, and are classified in the following : Unhealthy for Sensitive Groups Days, Unhealthy Days , Very Unhealthy 
    Days, and Hazardous Days. 'Total Unhealthy AQI days' are then divided by the total number of days with an AQI value. Final CRI
    metric includes one value per county from the data spanning 1980-2022.

    Parameters
    ----------
    input_folder: string
              Folder name containing EPA's summarized air quality index values (resulting folder from epa_air_quality_pull.py)
    filtered_csv_file: string
              Name of csv file containing CA county data
    final_output_name: string
              Name of the final csv file that contains CRI metric
    '''

    # Filter data to isolate 'State' header to California
    filtered_data = []
    filter_condition = "California"

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"): 
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    header = next(csv_reader)
                    state_index = header.index('State')
                    for row in csv_reader:
                        if row[state_index] == filter_condition:
                            filtered_data.append(row)
            else:
                print('file(s) not found')

    print('Deleting repeat years and creating single csv containing all California AQI data')
    print('')
    print(f'Data filtered to California Counties, file is called {filtered_csv_file}')

    # Create new single file containing all CA AQI data
    with open(filtered_csv_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        for row in filtered_data:
            csv_writer.writerow(row)

    # Eliminate repeat years with identical data (occurs for 1980 & 1981)
    ca_air_quality = pd.read_csv(filtered_csv_file)
    ca_air_quality = ca_air_quality.drop_duplicates(subset=['Year', 'County'])
    ca_air_quality.to_csv('natural_epa_air_quality.csv')

    # Create df that holds desired data variables
    columns_to_sum = ['Days with AQI', 
                      'Unhealthy for Sensitive Groups Days',
                      'Unhealthy Days',
                      'Very Unhealthy Days',
                      'Hazardous Days'
                      ]
    # Group data by county and sum desired columns for the temporal range of the dataset (1980-2022)
    ca_county_unhealthy_days = ca_air_quality.groupby('County')[columns_to_sum].sum().reset_index()
    
    print('Calculating total unhealthy days per county')
    print('')

    # Create new column counting total unhealthy air quality days
    ca_county_unhealthy_days['Total Unhealthy AQI Days'] = (
        ca_county_unhealthy_days['Unhealthy for Sensitive Groups Days'] + 
        ca_county_unhealthy_days['Unhealthy Days'] + 
        ca_county_unhealthy_days['Very Unhealthy Days'] + 
        ca_county_unhealthy_days['Hazardous Days']
    )

    print('Calculating CRI Metric (total unhealthy AQI days/total AQI days)')
    print('')

    # Calculate CRI metric
    ca_county_unhealthy_days['CRI Metric'] = (
        ca_county_unhealthy_days['Total Unhealthy AQI Days'] / ca_county_unhealthy_days['Days with AQI']
    )

    # Save data as a single .csv file, adding .csv string if not entered
    if not final_output_name.endswith('.csv'):
        final_output_name += '.csv'

    ca_county_unhealthy_days.to_csv(final_output_name)
    if os.path.isfile(final_output_name):
        print(f'Calculation made, .csv file named: {final_output_name} was created')
    else:
        print('Final .csv file containing calculation was not made')

process_air_quality_data('air_quality_csv_files', 'all_ca_air_quality_data', 'CRI_metric_air_quality_1980_2022')