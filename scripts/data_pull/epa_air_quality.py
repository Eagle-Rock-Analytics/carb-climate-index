#Load some library
import requests
from bs4 import BeautifulSoup
import csv
import zipfile
import io
from urllib.parse import urljoin

# Web scraping function that points to url and downloads links within the starting string variable.
# Downloaded zip files are extracted and converted to csv's
def scrape_website(base_url, starting_string, gathered_links, download_dir):
    '''
    Web scrapes EPA's annual summarized air quality index (AQI) data. The data is unzipped and moved into a desired folder as .csv 
    files. Data contains AQI data by county, and each file represents a year from 1980-2022.
    
    Parameters
    ----------
    base_url: string
              Use the url to EPA's annual summary AQI data: https://aqs.epa.gov/aqsweb/airdata/download_files.html#Annual
    starting_string: string
              A shared string that all links to the data within the url share: annual_aqi_by_county
    gathered_links: string
              Name a place to store all of the url references before data is converted to a folder with all the data
    download_dir: string
              Name of the folder which will hold data csv files
    '''
    
    scraped_links = []
    print('Searching web for url')
    print('')
    try:
        # Send an HTTP GET request to the specified base_url with SSL certificate verification disabled
        response = requests.get(base_url, verify=True)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            print('Url request sucessful!')
            print('')

            links = soup.find_all('a')
            for link in links:
                link_text = link.text
                link_href = link.get('href')
                if link_href and link_href.startswith(starting_string):
                    scraped_links.append((link_text, link_href))
            print(f'Data links found, placing into .csv file called: {gathered_links}')

            print('')

            # Write the scraped links to a CSV file
            with open('gathered_links.csv', 'w', newline='') as csv_file:

                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Link Text', 'Link Href'])  # Write header row
                for link in scraped_links:
                    csv_writer.writerow(link)
            print(f'File: {gathered_links} created')
            print('')

            # Download and extract data from ZIP files
            print(f'Extracting data and placing into folder: {download_dir}')
            print('')
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

    print(f'{download_dir} folder created and completed, data has been pulled')
 
 # Run the function to extract EPA's county level annual air quality data      
scrape_website('https://aqs.epa.gov/aqsweb/airdata/download_files.html#Annual', 'annual_aqi_by_county', 'gathered_links', 'air_quality_csv_files')