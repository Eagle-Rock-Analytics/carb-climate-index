import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import warnings

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
    df = _domain_plot_weighting(df, society, built, natural)
    
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

def _domain_plot_weighting(df, society, built, natural):
    '''In order to visualize the importance of weighting each domain'''
    df['DUMMY_society_tract_adjusted'] = df['DUMMY_society_tract_adjusted'] * society
    df['DUMMY_built_tract_adjusted'] = df['DUMMY_built_tract_adjusted'] * built
    df['DUMMY_natural_tract_adjusted'] = df['DUMMY_natural_tract_adjusted'] * natural
    return df

def plot_domain(gdf, domain, savefig=False):
    if domain == "society_":
        domain = "society_economy"

    # check for invalid geometries
    if len(gdf) == 0:
        print('No valid geometries. Cannot plot.')

    else:
        # Set up the figure
        fig, ax = plt.subplots(1, 1, figsize=(5, 8), layout='compressed')

        # Plot the data
        plot = gdf.plot(column=f'summed_indicators_{domain}_domain_min_max_standardized', 
                ax=ax, 
                vmin=0, vmax=1, 
                legend=True, 
                cmap='RdYlBu_r',
                legend_kwds={'label': 'Vulnerability (larger values are more vulnerable)', 'orientation': 'horizontal', 'shrink': 1.0, 'pad': 0.03})
        
        # Set title
        ax.set_title(f'Cal-CRAI: {domain} domain', fontsize = 16.5)

        # Display the plot
        plt.show()

        # export figure
        if savefig:
            figname = f'{domain}_domain_figure'
            fig.savefig(f'{figname}.png', format='png', dpi=300, bbox_inches='tight')
            print('Figure exported!')

def plot_region_domain(gdf, counties_to_plot, savefig=False):
    """
    Plots a domain score vulnerability for selected counties.

    Parameters:
    -----------
    gdf : GeoDataFrame
        A GeoDataFrame containing the data you want to plot, which must include the column 'GEOID' to match with the census tract data.
    
    counties_to_plot : list of str, optional
        A list of county FIPS codes (as strings) to plot. If None, no counties will be plotted.
        Example: ['037', '071', '065', '029', '111'].
    
    savefig : bool, optional
        If True, the plot will be saved as a PNG file. Default is False.

    Returns:
    --------
    None
        Displays the plot. Optionally saves the plot as a PNG file.
    
    Notes:
    ------
    - The function assumes the column 'summed_indicators_society_economy_domain_min_max_standardized'
      exists in the provided `gdf` and is the column to be visualized on the map.
    - The census tract data for California is loaded from an external source ('s3://ca-climate-index/0_map_data/...').
    - Invalid geometries (if any) will be reported, but they do not prevent the plot from being displayed.
    """
    
    
    # Dictionary of county labels
    county_labels = {
        '001': 'Alameda', '003': 'Alpine', '005': 'Amador', '007': 'Butte', '009': 'Calaveras',
        '011': 'Colusa', '013': 'Contra Costa', '015': 'Del Norte', '017': 'El Dorado', '019': 'Fresno',
        '021': 'Glenn', '023': 'Humboldt', '025': 'Imperial', '027': 'Inyo', '029': 'Kern',
        '031': 'Kings', '033': 'Lake', '035': 'Lassen', '037': 'Los \n Angeles', '039': 'Madera',
        '041': 'Marin', '043': 'Mariposa', '045': 'Mendocino', '047': 'Merced', '049': 'Modoc',
        '051': 'Mono', '053': 'Monterey', '055': 'Napa', '057': 'Nevada', '059': 'Orange',
        '061': 'Placer', '063': 'Plumas', '065': 'Riverside', '067': 'Sacramento', '069': 'San Benito',
        '071': 'San Bernardino', '073': 'San Diego', '075': 'San Francisco', '077': 'San Joaquin',
        '079': 'San Luis Obispo', '081': 'San Mateo', '083': 'Santa Barbara', '085': 'Santa Clara',
        '087': 'Santa Cruz', '089': 'Shasta', '091': 'Sierra', '093': 'Siskiyou', '095': 'Solano',
        '097': 'Sonoma', '099': 'Stanislaus', '101': 'Sutter', '103': 'Tehama', '105': 'Trinity',
        '107': 'Tulare', '109': 'Tuolumne', '111': 'Ventura', '113': 'Yolo', '115': 'Yuba'
    }

    # Example of counties you want to plot (you'll pass this in as the `counties_to_plot` parameter)
    # counties_to_plot = ['037', '071', '065', '029', '111']

    # Call in the census tract data
    census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/"
    ca_boundaries = gpd.read_file(census_shp_dir)
    ca_boundaries['GEOID'] = ca_boundaries['GEOID'].astype(str)
    
    # Merge the passed GeoDataFrame with the census boundary data
    df2 = gdf.merge(ca_boundaries, on='GEOID')

    # Filter rows where COUNTYFP is in the `counties_to_plot` list
    df2_filtered = df2[df2['COUNTYFP'].isin(counties_to_plot)]

    # Convert to GeoDataFrame with the correct CRS if necessary
    df2_filtered = gpd.GeoDataFrame(df2_filtered, geometry='geometry', crs=4269)

    # Check for invalid geometries
    invalid_geometries = df2_filtered[~df2_filtered['geometry'].is_valid]
    print("Number of invalid geometries:", len(invalid_geometries))

    # Group by COUNTYFP and take the geometry of the first row in each group
    county_boundaries = df2_filtered.dissolve(by='COUNTYFP')['geometry']

    # Check if there are any valid geometries left after filtering
    if len(county_boundaries) == 0:
        print('No valid geometries. Cannot plot.')
        return

    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 16), layout='compressed')

    # Plot county boundaries
    county_boundaries.boundary.plot(ax=ax, linewidth=0.55, edgecolor='black')

    # Plot the data
    df2_filtered.plot(column='summed_indicators_society_economy_domain_min_max_standardized', 
                      ax=ax, 
                      vmin=0, vmax=1, 
                      legend=True, 
                      cmap='RdYlBu_r', 
                      legend_kwds={'label': 'Vulnerability (larger values are more vulnerable)', 'orientation': 'horizontal', 'shrink': 0.9, 'pad': -0.3})


    # Suppress specific UserWarning messages
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS. Results from 'area' are likely incorrect.")
    
    # Find the min and max county area to scale the font sizes accordingly
    min_area = county_boundaries.area.min()
    max_area = county_boundaries.area.max()

    # Add county labels with dynamic font size based on normalized county area
    for county_code in counties_to_plot:
        label = county_labels.get(county_code, '')
        if label:  # Only add label if it exists in the dictionary
            # Get the centroid of the county
            centroid = county_boundaries[county_code].centroid

            # Dynamically adjust font size based on the area of the county
            county_area = county_boundaries[county_code].area

            # Normalize the area to a font size between 6 and 12
            if max_area > min_area:
                font_size = 6 + (12 - 6) * (county_area - min_area) / (max_area - min_area)
            else:
                font_size = 6  # In case all counties have the same area

            # Add text label to the plot
            ax.text(centroid.x, centroid.y, label, weight='roman', fontsize=font_size, ha='center', va='baseline')

        # Optionally save the figure
        if savefig:
            plt.savefig(f'region_plot_{counties_to_plot}.png', dpi=300)

    # Display the plot
    plt.show()

    