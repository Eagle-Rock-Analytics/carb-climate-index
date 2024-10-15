import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from shapely.geometry import box

census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/"
ca_boundaries = gpd.read_file(census_shp_dir)


# Dictionary mapping county codes to labels
def _id_county_label(county_code, label):
    county_labels = {
        '001': 'Alameda',
        '003': 'Alpine',
        '005': 'Amador',
        '007': 'Butte',
        '009': 'Calaveras',
        '011': 'Colusa',
        '013': 'Contra Costa',
        '015': 'Del Norte',
        '017': 'El Dorado',
        '019': 'Fresno',
        '021': 'Glenn',
        '023': 'Humboldt',
        '025': 'Imperial',
        '027': 'Inyo',
        '029': 'Kern',
        '031': 'Kings',
        '033': 'Lake',
        '035': 'Lassen',
        '037': 'Los \n Angeles',  # Split name for better label placement
        '039': 'Madera',
        '041': 'Marin',
        '043': 'Mariposa',
        '045': 'Mendocino',
        '047': 'Merced',
        '049': 'Modoc',
        '051': 'Mono',
        '053': 'Monterey',
        '055': 'Napa',
        '057': 'Nevada',
        '059': 'Orange',
        '061': 'Placer',
        '063': 'Plumas',
        '065': 'Riverside',
        '067': 'Sacramento',
        '069': 'San Benito',
        '071': 'San Bernardino',
        '073': 'San Diego',
        '075': 'San Francisco',
        '077': 'San Joaquin',
        '079': 'San Luis \n Obispo',  # Split name
        '081': 'San Mateo',
        '083': 'Santa Barbara',
        '085': 'Santa Clara',
        '087': 'Santa Cruz',
        '089': 'Shasta',
        '091': 'Sierra',
        '093': 'Siskiyou',
        '095': 'Solano',
        '097': 'Sonoma',
        '099': 'Stanislaus',
        '101': 'Sutter',
        '103': 'Tehama',
        '105': 'Trinity',
        '107': 'Tulare',
        '109': 'Tuolumne',
        '111': 'Ventura',
        '113': 'Yolo',
        '115': 'Yuba'
    }

def index_plot(df, scenario=None, save=False):
    '''Maps the Cal-CRAI index value for entire state'''

    # plotting help
    df2 = df.merge(ca_boundaries, on='GEOID')
    df2['geometry'] = df2['geometry_y']
    df2 = df2.drop(columns = ['geometry_x','geometry_y'])
    df2 = gpd.GeoDataFrame(df2, geometry='geometry', crs=4269)

    # set-up figure
    fig, ax = plt.subplots(1, 1, figsize=(4.5,6), layout='compressed')

    df2.plot(column='calcrai_score', ax=ax, vmin=-3, vmax=3, legend=True, cmap='RdYlBu',
                     legend_kwds={'label':'Cal-CRAI Index value', 'orientation': 'horizontal', 'shrink':0.7});

    if scenario == None:
        plt.annotate('Equal-weighted domains'.format(scenario), xy=(0.02, 0.02), xycoords='axes fraction')
    if scenario != None:
        plt.annotate('Weighting for {}'.format(scenario), xy=(0.02, 0.02), xycoords='axes fraction')

    if save:
        fig.savefig('dummy_ca_map.png', dpi=300, bbox_inches='tight') ## need to replace fig name once data repo completed

def index_domain_plot(df, scenario=None, society=1, built=1, natural=1, save=False):
    '''Produces subplots of the Cal-CRAI index value and the corresponding domains'''
    # internally weight domains
    df = domain_plot_weighting(df, society, built, natural)
    
    # plotting help
    df2 = df.merge(ca_boundaries, on='GEOID')
    df2['geometry'] = df2['geometry_y']
    df2 = df2.drop(columns = ['geometry_x','geometry_y'])
    df2 = gpd.GeoDataFrame(df2, geometry='geometry', crs=4269)

    # set-up figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(12,6), layout='compressed')
    vmin=-3
    vmax=3
    dmin=-1
    dmax=1
    cm='RdYlBu'

    df2.plot(ax=ax1, column='calcrai_score', vmin=vmin, vmax=vmax, legend=True, cmap=cm,
             legend_kwds={'label':'Cal-CRAI Index value', 'orientation': 'horizontal', 'shrink':0.7});
    df2.plot(ax=ax2, column='DUMMY_society_tract_adjusted', vmin=dmin, vmax=dmax, legend=True, cmap=cm,
             legend_kwds={'label':'Society & Economy \ndomain weight', 'orientation': 'horizontal', 'shrink':0.7});
    df2.plot(ax=ax3, column='DUMMY_natural_tract_adjusted', vmin=dmin, vmax=dmax, legend=True, cmap=cm,
             legend_kwds={'label':'Natural Systems \ndomain weight', 'orientation': 'horizontal', 'shrink':0.7});
    df2.plot(ax=ax4, column='DUMMY_built_tract_adjusted', vmin=dmin, vmax=dmax, legend=True, cmap=cm,
             legend_kwds={'label':'Built Environment \ndomain weight', 'orientation': 'horizontal', 'shrink':0.7});

    if scenario == None:
        ax1.annotate('Equal-weighted domains'.format(scenario), xy=(0.02, 0.02), xycoords='axes fraction')
    else: 
        ax1.annotate('Weighting for {}'.format(scenario), xy=(0.02, 0.02), xycoords='axes fraction')

    if save:
        fig.savefig('dummy_ca_domains_map.png', dpi=300, bbox_inches='tight') ## need to replace fig name once data repo completed

def domain_plot_weighting(df, society, built, natural):
    '''In order to visualize the importance of weighting each domain'''
    df['DUMMY_society_tract_adjusted'] = df['DUMMY_society_tract_adjusted'] * society
    df['DUMMY_built_tract_adjusted'] = df['DUMMY_built_tract_adjusted'] * built
    df['DUMMY_natural_tract_adjusted'] = df['DUMMY_natural_tract_adjusted'] * natural
    return df

def plot_domain(gdf, domain, savefig=False):
    # check for invalid geometries
    if len(gdf) == 0:
        print('No valid geometries. Cannot plot.')
    else:
        # Set up the figure
        fig, ax = plt.subplots(1, 1, figsize=(5, 8), layout='compressed')

        # Define the column to plot
        column_to_plot = f'summed_indicators_{domain}domain_min_max_standardized'

        # Check if the alternative column exists in the GeoDataFrame
        if 'loss_exposure_product_min_max_standardized' in gdf.columns:
            column_to_plot = 'loss_exposure_product_min_max_standardized'

        # Plot the data
        plot = gdf.plot(column=column_to_plot, 
                ax=ax, 
                vmin=0, vmax=1, 
                legend=True, 
                cmap='RdYlBu_r',
                legend_kwds={'label': 'Vulnerability (larger values are more vulnerable)', 'orientation': 'horizontal', 'shrink': 1.0, 'pad': 0.03})
        
        # Set title
        # Adjust the domain string to replace underscores with spaces and capitalize each word
        formatted_domain = domain.replace('_', ' ').title()

        # Set the plot title using the formatted domain string
        ax.set_title(f'Cal-CRAI: {formatted_domain} Domain', fontsize=16.5)

        # Display the plot
        plt.show()

        # export figure
        if savefig:
            figname = f'{domain}_domain_figure'
            fig.savefig(f'{figname}.png', format='png', dpi=300, bbox_inches='tight')
            print('Figure exported!')

def plot_region_domain(gdf, counties_to_plot=None, region=None, plot_all=False, savefig=False, font_color='black', domain='society_economy_', domain_label_map=None):
    """
    Plots a domain score vulnerability for selected counties or regions, with the option to exclude features within a bounding box.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        A GeoDataFrame containing the data you want to plot, which must include the column 'GEOID' to match with the census tract data.
    
    counties_to_plot : list of str, optional
        A list of county FIPS codes (as strings) to plot. If None, no counties will be plotted.
        Example: ['037', '071', '065', '029', '111'].
    
    region : str, optional
        A predefined region to plot. Options: 'bay_area', 'central_region', 'inland_deserts', 'north_central', 'northern', or 'south_coast'.
        If specified, this will override `counties_to_plot`.
    
    plot_all : bool, optional
        If True, plots all counties in California. Overrides `counties_to_plot` and `region`.
    
    savefig : bool, optional
        If True, the plot will be saved as a PNG file. Default is False.

    font_color : str, optional
        Color of the font for county labels. Default is 'black'.

    domain : str, optional
        The domain name used for labeling and column names. Default is 'society_economy_'.

    domain_label_map : dict, optional
        A dictionary to map the domain variable to a more readable label. Example: {'society_economy_': 'Society and Economy Domain'}
    
    Returns:
    --------
    None
        Displays the plot. Optionally saves the plot as a PNG file.
    """
    
    # If a domain label map is provided, use it to get a readable title. Otherwise, create it from the domain string.
    if domain_label_map:
        domain_name = domain_label_map.get(domain, domain.replace('_', ' ').title())
    else:
        domain_name = domain.replace('_', ' ').title()

    # Dictionary of county labels
    county_labels = {
        '001': 'Alameda', '003': 'Alpine', '005': 'Amador', '007': 'Butte', '009': 'Calaveras',
        '011': 'Colusa', '013': 'Contra Costa', '015': 'Del Norte', '017': 'El Dorado', '019': 'Fresno',
        '021': 'Glenn', '023': 'Humboldt', '025': 'Imperial', '027': 'Inyo', '029': 'Kern',
        '031': 'Kings', '033': 'Lake', '035': 'Lassen', '037': 'Los Angeles', '039': 'Madera',
        '041': 'Marin', '043': 'Mariposa', '045': 'Mendocino', '047': 'Merced', '049': 'Modoc',
        '051': 'Mono', '053': 'Monterey', '055': 'Napa', '057': 'Nevada', '059': 'Orange',
        '061': 'Placer', '063': 'Plumas', '065': 'Riverside', '067': 'Sacramento', '069': 'San Benito',
        '071': 'San Bernardino', '073': 'San Diego', '075': 'San Francisco', '077': 'San Joaquin',
        '079': 'San Luis Obispo', '081': 'San Mateo', '083': 'Santa Barbara', '085': 'Santa Clara',
        '087': 'Santa Cruz', '089': 'Shasta', '091': 'Sierra', '093': 'Siskiyou', '095': 'Solano',
        '097': 'Sonoma', '099': 'Stanislaus', '101': 'Sutter', '103': 'Tehama', '105': 'Trinity',
        '107': 'Tulare', '109': 'Tuolumne', '111': 'Ventura', '113': 'Yolo', '115': 'Yuba'
    }

    # Define the new regional groups of counties
    regions = {
        'bay_area': ['001', '013', '041', '055', '081', '085', '087', '075', '095', '097'],
        'central_region': ['019', '029', '031', '039', '043', '047', '053', '069', '079', '099', '107', '109'],
        'inland_deserts': ['025', '027', '051', '065', '071'],
        'north_central': ['067', '077', '017', '033', '057', '061', '091', '101', '063', '113', '115'],
        'northern': ['015', '023', '035', '045', '049', '093', '089', '103', '105'],
        'south_coast': ['037', '059', '073', '083', '111'],
        'slr_coast' : ['001', '013', '015', '023', '037', '041', '045', '053', '055', '059', '067', '073', '075', '077', '079', '081', '083', '085', '087', '095', '097', '111', '113']
    }

    # Set counties_to_plot based on the specified region or plot_all flag
    if plot_all:
        counties_to_plot = list(county_labels.keys())
        title = f'Vulnerability Index of All Counties in California - {domain_name}'
    elif region:
        counties_to_plot = regions.get(region, [])
        region_name = region.replace('_', ' ').title()  # Capitalize the region name for display
        title = f'Vulnerability Index of California\'s {region_name} - {domain_name}'
    else:
        title = f'Vulnerability Index of Selected Counties \n {domain_name}'

    # Load the census tract data
    census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/"
    ca_boundaries = gpd.read_file(census_shp_dir)
    ca_boundaries['GEOID'] = ca_boundaries['GEOID'].astype(str)
    
    # Merge the passed GeoDataFrame with the census boundary data
    df2 = gdf.merge(ca_boundaries, on='GEOID')

    # Filter rows where COUNTYFP is in the `counties_to_plot` list
    df2_filtered = df2[df2['COUNTYFP'].isin(counties_to_plot)]

    # Convert to GeoDataFrame with the correct CRS if necessary
    df2_filtered = gpd.GeoDataFrame(df2_filtered, geometry='geometry', crs=4269)

    # Define the bounding box to exclude (xmin, ymin, xmax, ymax)
    exclusion_box = box(-122.8, 37.6, -123.2, 37.85) 
    
    # Exclude features within the bounding box
    df2_filtered = df2_filtered[~df2_filtered.intersects(exclusion_box)]

    # Check for invalid geometries
    invalid_geometries = df2_filtered[~df2_filtered['geometry'].is_valid]
    print("Number of invalid geometries:", len(invalid_geometries))

    # Group by COUNTYFP and take the geometry of the first row in each group
    county_boundaries = df2_filtered.dissolve(by='COUNTYFP')['geometry']

    # Check if there are any valid geometries left after filtering
    if len(county_boundaries) == 0:
        print('No valid geometries. Cannot plot.')
        return

    # Adjust figure size and padding based on the type and number of counties/regions
    if plot_all:
        fig_size = (8, 18)
        base_font_size = 5
    elif region:
        fig_size = (8, 14)
        base_font_size = 8
    else:
        # Specific counties
        num_counties = len(counties_to_plot)
        if num_counties == 1:
            fig_size = (6, 6)
            base_font_size = 12
        else:
            fig_size = (6 + (num_counties - 1) // 2, 10 + ((num_counties - 1) // 2) * 2)
            base_font_size = min(10, 6 + (num_counties / 10))

    fig, ax = plt.subplots(1, 1, figsize=fig_size, layout='compressed')

    # Plot county boundaries
    county_boundaries.boundary.plot(ax=ax, linewidth=0.55, edgecolor='black')

    # Define the column to plot
    column_to_plot = f'summed_indicators_{domain}domain_min_max_standardized'

    # Check if the alternative column exists in the GeoDataFrame
    if 'loss_exposure_product_min_max_standardized' in gdf.columns:
        column_to_plot = 'loss_exposure_product_min_max_standardized'

    # Plot the data
    df2_filtered.plot(column=column_to_plot, 
                      ax=ax, 
                      vmin=0, vmax=1, 
                      legend=True, 
                      cmap='RdYlBu_r', 
                      legend_kwds={'label': 'Vulnerability (larger values are more vulnerable)', 'orientation': 'horizontal', 'shrink': 0.9})

    # Suppress specific UserWarning messages
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS. Results from 'area' are likely incorrect.")
    
    # Find the min and max county area to scale the font sizes accordingly
    min_area = county_boundaries.area.min()
    max_area = county_boundaries.area.max()

    # Adjust font size based on figure size and county area
    for county_code in counties_to_plot:
        label = county_labels.get(county_code, '')
        if label:  # Only add label if it exists in the dictionary
            # Get the centroid of the county
            centroid = county_boundaries[county_code].centroid

            # Dynamically adjust font size based on the area of the county
            county_area = county_boundaries[county_code].area

            # Normalize the area to a font size
            if max_area > min_area:
                font_size = base_font_size + (10 - base_font_size) * (county_area - min_area) / (max_area - min_area)
            else:
                font_size = base_font_size

            # Add text label to the plot with specified font color
            ax.text(centroid.x, centroid.y, label, weight='medium', fontsize=font_size, color=font_color, ha='center', va='baseline', alpha=1)

    # Set the plot title
    ax.set_title(title, fontsize=13, weight='normal')

    # Automatically adjust padding to be below x-axis ticks
    x_ticks = ax.get_xticks()
    x_tick_labels = ax.get_xticklabels()
    max_label_height = max([tick.get_window_extent().height for tick in x_tick_labels])

    # Adjust padding based on the maximum label height
    padding = max_label_height / fig.dpi

    # Optionally save the figure
    if savefig:
        plt.savefig(f'region_plot_{counties_to_plot}.png', dpi=300)

    # Display the plot
    plt.show()

    