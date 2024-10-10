# List of columns that have a few missing values
COLUMNS_WITH_FEW_MISSING = [
    'distance_fire_stations', 'distance_rivers', 'distance_roads', 'distance_powerlines',
    'cropland', 'forest_deciduous_broad', 'forest_deciduous_needle', 'forest_evergreen_broad', 
    'forest_evergreen_needle', 'forest_mixed', 'forest_unknown', 'herbaceous_vegetation', 
    'moss_lichen', 'shrubland', 'sprarse_vegetation', 'urban', 'water', 'wetland'
]

# List of vegetation columns to be validated
VEGETATION_COLS = [
    'cropland', 'forest_deciduous_broad', 'forest_deciduous_needle', 'forest_evergreen_broad',
    'forest_evergreen_needle', 'forest_mixed', 'forest_unknown', 'herbaceous_vegetation',
    'moss_lichen', 'shrubland', 'sprarse_vegetation', 'urban', 'water', 'wetland'
]

# List of features of interest after EDA
FEATURES_OF_INTEREST = ['max_wind_vel', 'urban', 'cropland', 
    'forest_deciduous_broad', 'forest_evergreen_broad', 'forest', 'ignition'
]

# List of features to apply PCA
PCA_COLS = ['distance_roads', 'distance_rivers', 'distance_powerlines', 'distance_fire_stations'
]

# List of key columns to remain at last
KEY_COLUMNS = ['aspect','elevation','pop_dens','slope',
 'anom_max_temp','anom_max_wind_vel','anom_avg_temp','anom_avg_rel_hum','anom_avg_soil','anom_sum_prec',
 'max_temp','max_wind_vel','avg_temp','avg_wind_angle','avg_rel_hum','avg_soil','sum_prec','forest','max_max_temp',
 'wind_forest_interaction','urban_proximity','heat_index','cumulative_distance','pca_dist_1','pca_dist_2','ignition'
]
