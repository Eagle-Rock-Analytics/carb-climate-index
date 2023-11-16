#Load some library
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import zipfile
import io
from urllib.parse import urljoin
import os

# Web scraping function that points to url and downloads links within the starting string variable.
# Downloaded zip files are extracted and converted to csv's
def scrape_website(base_url, starting_string, csv_filename, download_dir):
    scraped_links = []
    try:
        # Send an HTTP GET request to the specified base_url with SSL certificate verification disabled
        response = requests.get(base_url, verify=True)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

           
            links = soup.find_all('a')
            for link in links:
                link_text = link.text
                link_href = link.get('href')
                if link_href and link_href.startswith(starting_string):
                    scraped_links.append((link_text, link_href))

            # Write the scraped links to a CSV file
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Link Text', 'Link Href'])  # Write header row
                for link in scraped_links:
                    csv_writer.writerow(link)

            # Download and extract data from ZIP files
            for link_text, link_href in scraped_links:
                zip_url = urljoin(base_url, link_href)  # Correctly construct the URL
                response = requests.get(zip_url)
                if response.status_code == 200:
                    zip_data = response.content
                    try:
                        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
                            zip_ref.extractall(download_dir)
                    except zipfile.BadZipFile:
                        print(f"Failed to extract: {link_text} ({link_href}) is not a valid ZIP file.")

        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

 # Run the function to extract EPA's county level annual air quality data      
scrape_website('https://aqs.epa.gov/aqsweb/airdata/download_files.html#Annual', 'annual_aqi_by_county', 'annual_aqi_by_county', 'downloaded_data_dir')

# Directory where the CSV files are located
input_folder = "downloaded_data_dir"

# Output CSV file to store the filtered data
output_csv_file = "all_ca_air_quality_data.csv"

# Initialize an empty list to store the filtered data
filtered_data = []

# Define the filter condition (filtering files for California)
filter_condition = "California"

# Iterate through CSV files in the input folder
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".csv"): 
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                header = next(csv_reader)  # Read the header row
                state_index = header.index('State')  # Find the index of 'State' column
                for row in csv_reader:
                    if row[state_index] == filter_condition:
                        filtered_data.append(row)

# Write the filtered data to a single CSV file
with open(output_csv_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)  # Write the header row
    for row in filtered_data:
        csv_writer.writerow(row)
        
# Read in new CSV file with all CA county data for all years
ca_air_quality = pd.read_csv('all_ca_air_quality_data.csv')

# Saw this dataset had repeat data for two years
# Drop duplicate years per county
ca_air_quality = ca_air_quality.drop_duplicates(subset=['Year', 'County'])
ca_air_quality.to_csv('natural_epa_air_quality.csv') #saving CA air quality data that has been cleaned of repeats

# Group by 'county' and calculate the sum of each columns values between all years
columns_to_sum = ['Days with AQI', 'Unhealthy for Sensitive Groups Days', 'Unhealthy Days', 'Very Unhealthy Days', 'Hazardous Days']
ca_county_unhealthy_days = ca_air_quality.groupby('County')[columns_to_sum].sum().reset_index()

# New column to add all unhealthy air variables into one total
ca_county_unhealthy_days['Total Unhealthy Days'] = ca_county_unhealthy_days['Unhealthy for Sensitive Groups Days'] + ca_county_unhealthy_days['Unhealthy Days'] + ca_county_unhealthy_days['Very Unhealthy Days'] + ca_county_unhealthy_days['Hazardous Days']

# Calculating desired CRI metric by dividing new total unhealthy days variable by the total number of days testing AQI
ca_county_unhealthy_days['CRI Metric'] = ca_county_unhealthy_days['Total Unhealthy Days'] / ca_county_unhealthy_days['Days with AQI']


# Saving CRI metric data as a CSV
ca_county_unhealthy_days.to_csv('CRI_metric_air_quality_1980_2022.csv')
