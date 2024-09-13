import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/"
ca_boundaries = gpd.read_file(census_shp_dir)


# Dictionary mapping county codes to labels
# expand this further for all
def _id_county_label(county_code, label):
    county_labels = {
        '037': 'Los \n Angeles',
        '071': 'San Bernardino',
        '065': 'Riverside',
        '029': 'Kern',
        '111': 'Ventura'
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

def plot_region_domain(gdf, counties_to_viz, domain, savefig=False):
    if domain == "society_":
        domain = "society_economy"

    # subset for counties_to_viz
    gdf_filter = gdf[gdf['COUNTYFP'].isin(counties_to_viz)]

    # set-up figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 8), layout='compressed')

    # Plot county boundaries
    subset_boundaries = ca_boundaries[ca_boundaries['COUNTYFP'].isin(counties_to_viz)]
    subset_boundaries.boundary.plot(ax=ax, linewidth=0.7, edgecolor='black')

    # Plot the data
    gdf_filter.plot(column=f'summed_indicators_{domain}_domain_min_max_standardized', 
            ax=ax, 
            vmin=0, vmax=1, 
            legend=True, 
            cmap='RdYlBu_r', 
            legend_kwds={'label': 'Vulnerability (larger values are more vulnerable)', 'orientation': 'horizontal', 'shrink': 0.9, 'pad': -0.1})

    ## Add county labels
    # for county_code, label in county_labels.items():
    #     centroid = ca_boundaries[county_code].centroid
    #     ax.text(centroid.x, centroid.y, label, weight='light', fontsize=9, ha='center', va='baseline')

    ax.set_title(f'Cal-CRAI: {domain}, \nRegion of Interest', fontsize=16)

    # Display the plot
    plt.show()

    # export figure
    if savefig:
        figname = f'{domain}_domain_figure_subset'
        fig.savefig(f'{figname}.png', format='png', dpi=300, bbox_inches='tight')
        print('Figure exported!')

    