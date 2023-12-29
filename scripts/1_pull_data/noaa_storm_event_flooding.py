import os
import csv

# Path to the main directory containing subfolders and CSV files
main_folder = 'noaa_storm_event_files'

# Output file to store the filtered and merged data
output_file = 'noaa_storm_events_ca.csv'

# Function to merge CSV files, filter data for California, and use headers from the first file
def merge_and_filter_california(folder_path, output):
    headers_written = False  # Flag to track if headers have been written to the output file

    # Open the output file in write mode
    with open(output, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Iterate through the directory structure
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Check if the file is a CSV file
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    # Open and read each CSV file
                    with open(file_path, 'r', newline='') as infile:
                        reader = csv.reader(infile)
                        # Read the header row from the first file and write it to the output file
                        if not headers_written:
                            headers = next(reader)
                            writer.writerow(headers)
                            headers_written = True
                            
                            # Find the index of the column related to 'State'
                            state_index = headers.index('STATE')
                            
                        # Write rows for California only using the index of the 'State' column
                        for row in reader:
                            if headers_written and row[state_index] == 'CALIFORNIA':  # Filter for California data
                                writer.writerow(row)

# Call the function with the main folder path and output file
merge_and_filter_california(main_folder, output_file)