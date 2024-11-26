import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from shapely.geometry import box # type: ignore

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
 
def index_plot(df, column, scenario=None, plot_title=False, save_name=None, plot_type='continuous', vmin=-3, vmax=3):
    '''
    Maps the Cal-CRAI index for the entire state, can do descrete or continuous mapping
    depending on input column. 
    
    Parameters
    ----------
    df : DataFrame
        input Dataframe
    column : str
        Cal-CRAI column
    scenario : str
        Default is None. If index column is a weighted value, the user can input
        the name of the scenario to populate the figure title and annotation
    plot_title : bool
        Default is False, if using a scenario and want a title, set to True
    save_name : str
        Default is None, user can enter any string to save the figure as.
    plot_type : str
        Specifies the type of mapping for the plot. 
        - 'continuous': Uses a gradient to represent a smooth range of values.
        - 'discrete': Uses distinct colors to represent the binned data.
        Default is 'continuous'.
    vmin : int
        if plot is continuous, set the minimum bounds of the color gradient
        Default is -3
    vmax : int
        if plot is continuous, set the maximum bounds of the color gradient    
        Default is 3
    '''
    # Merging with geographical boundaries
    df2 = df.merge(ca_boundaries, on='GEOID')
    df2['geometry'] = df2['geometry']
    df2 = gpd.GeoDataFrame(df2, geometry='geometry', crs=4269)

    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 6), layout='compressed')

    # Check plot type and set plotting parameters accordingly
    if plot_type == 'discrete':
        # For discrete values (1-5), use discrete colormap
        df2.plot(column=column, ax=ax, legend=True, cmap='YlGnBu', categorical=True)
        ax.get_legend().set_title("Cal-CRAI Resiliency Percentiles")
    else:
        # For continuous values, use continuous colormap
        sm = df2.plot(column=column, ax=ax, vmin=vmin, vmax=vmax, cmap='bwr_r', legend=False)

        # Create a colorbar manually and set the title
        cbar = fig.colorbar(sm.collections[0], ax=ax, orientation='horizontal')
        cbar.set_label("Cal-CRAI Index Value")

    # Annotation for scenario
    if scenario is None:
        plt.annotate('Equal-weighted domains', xy=(0.02, 0.02), xycoords='axes fraction')
    else:
        plt.annotate('Weighting for {}'.format(scenario), xy=(0.02, 0.02), xycoords='axes fraction')
        if plot_title == True:
                ax.set_title(f'Cal-CRAI: {scenario.title()} Scenario', fontsize=16.5)

    # Save figure if required
    if save_name:
        fig.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')  # Save the figure

    plt.show()  # Show the plot

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
    cm='YlGnBu'

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

def plot_climate_domains(df, column_to_plot, domain='', savefig=False):
    '''
    Maps climate domain scores that contain the product values from summed risk and 
    exposure indicators per climate risk scenario.
    
    Parameters
    ----------
    df : DataFrame
        input Dataframe
    column_to_plot : str
        df's climate domain score column
    domain : str
        domain name, will go as the figre title
    savefig : bool
        if True, saves figure using the domain name as the save name
        Default is False
    '''
    # Merging with geographical boundaries
    df2 = df.merge(ca_boundaries, on='GEOID')
    df2['geometry'] = df2['geometry']
    df2 = gpd.GeoDataFrame(df2, geometry='geometry', crs=4269)

    # Check for invalid geometries
    if len(df2) == 0:
        print('No valid geometries. Cannot plot.')
    else:
        # Set up the figure
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 6), layout='compressed')

        # Plot the data
        df2.plot(
            column=column_to_plot,
            ax=ax,
            vmin=0, vmax=1,
            cmap='Blues'
        )

        # Create an inset axis for the colorbar inside the plot area (horizontal)
        cbar_ax = ax.inset_axes([0.48, 0.92, 0.48, 0.03])  # [x, y, width, height]
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []  # Required for the ScalarMappable
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Resilience', fontsize=10)

        # Set title
        formatted_domain = domain.replace('_', ' ').title()
        ax.set_title(f'{formatted_domain} Risk Domain', fontsize=16.5)

        # Display the plot
        plt.show()

        # Export figure
        if savefig:
            figname = f'{domain}_domain_figure'
            fig.savefig(f'{figname}.png', format='png', dpi=300, bbox_inches='tight')
            print('Figure exported!')

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
        if 'all_domain_loss_exposure_product_min_max_standardized' in gdf.columns:
            column_to_plot = 'all_domain_loss_exposure_product_min_max_standardized'

        # Plot the data
        plot = gdf.plot(column=column_to_plot, 
                ax=ax, 
                vmin=0, vmax=1, 
                legend=True, 
                cmap='Greens',
                legend_kwds={'label': 'Resilience (larger values are more resilient)', 'orientation': 'horizontal', 'shrink': 1.0, 'pad': 0.03})
        
        # Set title
        # Adjust the domain string to replace underscores with spaces and capitalize each word
        formatted_domain = domain.replace('_', ' ').title()

        # Set the plot title using the formatted domain string
        ax.set_title(f'Cal-CRAI: {formatted_domain}Domain', fontsize=16.5)

        # Display the plot
        plt.show()

        # export figure
        if savefig:
            figname = f'{domain}_domain_figure'
            fig.savefig(f'{figname}.png', format='png', dpi=300, bbox_inches='tight')
            print('Figure exported!')

def plot_region_domain(gdf, 
                       counties_to_plot=None,
                       region=None, plot_all=False,
                       savefig=False, font_color='black',
                       domain='society_economy_',
                       domain_label_map=None, 
                       vmin=0, vmax=1, 
                       column_to_plot=None,
                       cmap = 'Greens',
                       intro_title = 'Resiliency Index'):
    """
    Plots a domain score resilience for selected counties or regions, with the option to exclude features within a bounding box.
    
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
        if domain_name == '':
            title = f'{intro_title} of All Counties in California'
        else:
            title = f'{intro_title} of All Counties in California - {domain_name}'
        
    elif region:
        counties_to_plot = regions.get(region, [])
        region_name = region.replace('_', ' ').title()  # Capitalize the region name for display
        
        if domain_name == '':
            title = f'{intro_title} of California\'s {region_name}'
        else:
            title = f'{intro_title} of California\'s {region_name} - {domain_name}'

    else:
        title = f'{intro_title} of Selected Counties \n {domain_name}'

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

    fig, ax = plt.subplots(1, 1, figsize=fig_size, layout='constrained')

    # Plot county boundaries
    county_boundaries.boundary.plot(ax=ax, linewidth=0.55, edgecolor='black')

    # Define the column to plot
    if column_to_plot == None:
        column_to_plot = f'summed_indicators_{domain}domain_min_max_standardized'
    else:
        column_to_plot = column_to_plot

    # Plot the data
    df2_filtered.plot(column=column_to_plot, 
                      ax=ax, 
                      vmin=vmin, vmax=vmax, 
                      legend=True, 
                      cmap=cmap, 
                      legend_kwds={'label': f'{intro_title} (larger values are more resilient)', 'orientation': 'horizontal', 'shrink': 0.9,
                                   'shrink': 1.0, 'pad': 0.04})

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

    def index_weighting_plot_ranking(df, ca_boundaries, save=False):
        '''
        Creates ranking plots (1-5) with distinct colors for each scenario in the provided dataframe.
        '''
        # Automatically detect score columns by excluding 'GEOID'
        score_columns = [col for col in df.columns if col != 'GEOID']

        # Ensure score columns are numeric, converting errors to NaN
        df[score_columns] = df[score_columns].apply(pd.to_numeric, errors='coerce')

        # Fill NaN values with -inf to ensure they are ranked lowest
        df[score_columns] = df[score_columns].fillna(-np.inf)

        # Generate a color map for the scenarios
        scenario_colors = dict(zip(score_columns, plt.cm.Set1.colors[:len(score_columns)]))

        # Prepare the dataframe for ranking by calculating ranks within each row
        rank_df = df[['GEOID']].copy()
        rank_df[score_columns] = df[score_columns]

        # Sort each row’s values and get the top 5 columns for each GEOID
        ranked_columns = rank_df[score_columns].apply(lambda x: x.nlargest(5).index.tolist(), axis=1)
        ranked_scores = rank_df[score_columns].apply(lambda x: x.nlargest(5).tolist(), axis=1)
            
        # Create columns for each rank's scenario and score
        for i in range(5):
            rank_df.loc[:, f'rank_{i+1}_scenario'] = [col[i] for col in ranked_columns]
            rank_df.loc[:, f'rank_{i+1}_score'] = [score[i] for score in ranked_scores]

        # Merge with California boundaries for plotting
        df2 = rank_df.merge(ca_boundaries, on='GEOID')
        df2 = gpd.GeoDataFrame(df2, geometry='geometry', crs=4269)

        # Generate and save ranking plots (1-5)
        for rank in range(1, 6):
            fig, ax = plt.subplots(1, 1, figsize=(6, 8))

            # Prepare a column that assigns color based on the scenario for the current rank
            df2['color'] = df2[f'rank_{rank}_scenario'].map(scenario_colors)

            # Plot each geometry with its assigned color, suppressing the legend
            df2.plot(ax=ax, color=df2['color'], edgecolor='k', legend=False)

            # Manually create custom legend to avoid unwanted entries
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8)
                    for color in scenario_colors.values()]
            labels = scenario_colors.keys()

            # Only add a legend if there are labels to show
            if labels:
                ax.legend(handles, labels, title=f'Rank {rank} Scenario', loc='upper left')

            plt.annotate(f'Rank {rank} Cal-CRAI Index by Scenario', xy=(0.02, 0.02), xycoords='axes fraction')

            # Save the figure if required
            if save:
                fig.savefig(f'rank_{rank}_crai_index.png', dpi=300, bbox_inches='tight')

        plt.show()