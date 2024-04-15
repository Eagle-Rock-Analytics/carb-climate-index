def weight_domains(df, society, built, natural):
    '''
    Calculates the weighting scheme, based on input parameters:
    society, built, and natural
    '''
    governance_col = 'DUMMY_governance_summed_indicators_min_max_standardized'
    society_adjusted_col = 'DUMMY_society_tract_adjusted'
    built_adjusted_col = 'DUMMY_built_tract_adjusted'
    natural_adjusted_col = 'DUMMY_natural_tract_adjusted' 

    weighting = (
        df[governance_col] + 
        (society * (df[society_adjusted_col] * df[governance_col])) +
        (built * (df[built_adjusted_col] * df[governance_col])) +
        (natural * (df[natural_adjusted_col] * df[governance_col]))
    )

    df['calcrai_weighted'] = weighting
    return df


def calculate_index(df):
    '''Calcutes the Cal-CRAI index'''
    
    df['calcrai_score'] = df['calcrai_weighted'] / df['acute_risk']

    # testing for 0 values --> divide error
    df.loc[df['acute_risk'] == 0, 'calcrai_score'] = 0
    
    return df


def format_df(df):
    '''
    Minor clean-up of pandas df -- can be resolved in future version
    Demo purposes only, at present
    '''

    # df = df.drop(columns='field_1') # drops extra field
    df['GEOID'] = '0' + df['GEOID'] # formats GEOID column to match shapefile (has an extra 0 in front)

    for i in df.columns:
        exclude = ["geometry", "GEOID"]
        if i not in exclude:
            df[i] = df[i].astype(float) # changes type of core columns to float from string

    return df