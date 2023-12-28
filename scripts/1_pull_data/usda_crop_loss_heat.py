import os
import zipfile
import csv
from datetime import datetime

# Function to extract and read text files from zip files
def extract_and_read(file_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        file_contents = []
        for file in file_list:
            with zip_ref.open(file) as txt_file:
                lines = txt_file.readlines()
                lines = [line.decode('utf-8').strip() for line in lines]
                file_contents.append((file, lines))
    return file_contents

# Function to sort files based on the first column (assuming years)
def sort_files(file_contents):
    sorted_files = []
    for file, lines in file_contents:
        lines.sort(key=lambda x: get_year_from_line(x))
        sorted_files.append((file, lines))
    return sorted_files

# Function to extract year from line
def get_year_from_line(line):
    # Year is separated by '|' and is the first element
    elements = line.split('|')
    if len(elements) > 1:
        potential_year = elements[0].strip()
        if len(potential_year) == 4 and potential_year.isdigit():
            return potential_year
    return ''

# Function to merge files and write to a single CSV file, isolating rows with 'CA' in the third column
def merge_files(sorted_files, output_file):
    headers = ['year', 'state_code', 'state_abbreviation', 'county_code', 'county_name', 'commodity_code', 'commodity_name', 'insurance_plan_code', 'insurance_plan_abbreviation', 'stage_code', 'damage_cause_code', 'damage_description', 'determined_acres', 'indemnity_amount']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as merged_csv:
        csv_writer = csv.writer(merged_csv)
        csv_writer.writerow(['File'])  # Writing a row to denote the file name as a header
        csv_writer.writerow(headers)  # Writing headers
        
        for file, lines in sorted_files:
            for line in lines:
                row_data = [element.strip() for element in line.split('|')]
                if len(row_data) > 2 and row_data[2] == 'CA':  # Filtering rows with 'CA' in the third column
                    csv_writer.writerow(row_data)

# Main function
def main(folder_name, output_file):
    file_contents = []
    for file in os.listdir(folder_name):
        if file.endswith('.zip'):
            file_path = os.path.join(folder_name, file)
            file_contents.extend(extract_and_read(file_path))
    
    sorted_files = sort_files(file_contents)
    merge_files(sorted_files, output_file)
    print(f"Merged and sorted files written to {output_file}")

# Provide folder path containing zip files and output file name
# Zip files were downloaded and placed in a folder within our local env
folder_name = 'usda_crop_loss_heat_files'
output_file = 'usda_crop_loss_CA.csv'

# Run the main function
main(folder_name, output_file)
