
## Uncomment the lines below to install libraries only if required
# !pip install ee geemap pycrs gdown
# !pip install -U geemap

# import libraries
import ee
import geemap
import gdown
import os
import duckdb
import pandas as pd
import geopandas as gpd
import requests

# Connect to DuckDB and load spatial extension
con = duckdb.connect(database='uhi_test.duckdb')
con.execute("INSTALL spatial;")
con.execute("LOAD spatial;")


## Drop all current tables from the db so dealing with clean db
# Get a list of current table names
tables = con.execute("SHOW TABLES").fetchdf()['name'].tolist()

# Drop each table from the list
for table in tables:
    con.execute(f"DROP TABLE IF EXISTS {table}")
    print(f"Dropped table: {table}")

# # Uncomment only to run download of OS NGD building data locally
# Read building footprint data, this is a local path as the full dataset is too large to process remotely (3.26gb)
# Data source for OS NGD data for created London is hosted here:
# https://drive.google.com/file/d/198W9cWkrPN5NJvDk5khNvjiXqUebucb5/view?usp=drive_link
# download the data locally and then process

# Google Drive file ID (from the URL above)
file_id = '198W9cWkrPN5NJvDk5khNvjiXqUebucb5'

# Destination path to save the downloaded CSV
output_path = 'data/bld_fts_building/bld_fts_building.csv'

# create the output directory
os.makedirs('data/bld_fts_building', exist_ok=True)

# Construct the download URL
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file
gdown.download(url, output_path, quiet=False)


## Process buildings csv into DuckDB
# File path to CSV
csv_file_path = 'data/bld_fts_building/bld_fts_building.csv'
# Chunked loading with spatial geometry handling
chunk_size = 50000
csv_iterator = pd.read_csv(csv_file_path, chunksize=chunk_size)
# Name of the DuckDB table
table_name = 'buildings'
# iterate through chunks to load data into duckdb
for i, chunk in enumerate(csv_iterator):
    # Ensure geometry is treated as WKT (string) and create a spatial column
    chunk['geom'] = chunk['geometry'].apply(lambda wkt: f"ST_GeomFromText('{wkt}')" if pd.notnull(wkt) else 'NULL')
    # Write the chunk to DuckDB
    if i == 0:
        # Create the table from the first chunk, including geometry
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} AS
            SELECT *, ST_GeomFromText(geometry) AS geom
            FROM chunk
        """)
    else:
        # Insert subsequent chunks
        con.execute(f"""
            INSERT INTO {table_name}
            SELECT *, ST_GeomFromText(geometry) AS geom
            FROM chunk
        """)
# Add height and volume columns and calculate values
con.execute("""
    ALTER TABLE buildings ADD COLUMN height DOUBLE;
    ALTER TABLE buildings ADD COLUMN volume DOUBLE;

    UPDATE buildings
    SET height = COALESCE(height_relativeroofbase_m, height_absolutemin_m, 3),
        volume = geometry_area_m2 * COALESCE(height_relativeroofbase_m, height_absolutemin_m, 3);
""")

# create a clean version of the table
# select columns to keep
cols_to_keep = ['osid', 'height', 'volume', 'geometry_area_m2', 'geometry']
clean_cols = ', '.join(cols_to_keep)

# create clean table retaining only the columns specified
con.execute(f"""
    CREATE TABLE buildings_clean AS
    SELECT {clean_cols}
    FROM buildings;
""")

# Download boundary datasets for further processing
# this is prone to failure but data can be manually downloaded from:
# https://drive.google.com/drive/folders/1bXA6EpVIlHhWYg_tOtORb0S4BVIAL4Uj?usp=sharing

# # Google Drive folder ID
# folder_id = '1bXA6EpVIlHhWYg_tOtORb0S4BVIAL4Uj'

# # Destination path to save the downloaded folder
# output_path = 'data/'

# # Create the output directory if it doesn't exist
# os.makedirs(output_path, exist_ok=True)

# # Construct the folder download URL
# url = f'https://drive.google.com/drive/folders/{folder_id}'

# # Download the entire folder
# gdown.download_folder(url, output=output_path, quiet=False, use_cookies=False)


# set file paths from downloaded boundaries data
london_lsoa = 'data/boundaries/london_lsoa.shp'
boroughs = 'data/boundaries/London_Borough_Excluding_MHW.shp'

# drop the table first to create a clean version
con.execute(f"DROP TABLE IF EXISTS london_lsoa")
con.execute(f"DROP TABLE IF EXISTS london_boroughs")

# create london lsoa table
con.execute(f"""
    CREATE TABLE london_lsoa AS
    SELECT *
    FROM ST_Read('{london_lsoa}')
""")

# create london borough table
con.execute(f"""
    CREATE TABLE london_boroughs AS
    SELECT NAME, GSS_CODE, geom
    FROM ST_Read('{boroughs}')
""")


# add a column stating the borough of each LSOA
con.execute("""
    ALTER TABLE london_lsoa ADD COLUMN Borough TEXT;
    
    UPDATE london_lsoa
    SET Borough = SUBSTR(LSOA21NM, 1, LENGTH(LSOA21NM) - 5);
""")


# # Process total building volumes per lsoa
# This is split into batches to prevent time out issues.

# Get list of all LSOA codes
lsoa_codes = con.execute("""
    SELECT DISTINCT LSOA21CD
    FROM london_lsoa;
    """).fetchdf()['LSOA21CD'].tolist()

# Split into batches
batch_size = 250
batches = [lsoa_codes[i:i + batch_size] for i in range(0, len(lsoa_codes), batch_size)]

# Create the table before the loop if it doesn't exist
con.execute("""
    CREATE TABLE IF NOT EXISTS london_LSOAs_vol (
        LSOA21CD VARCHAR PRIMARY KEY,
        borough VARCHAR,
        building_count INTEGER,
        total_volume DOUBLE,
        geom GEOMETRY
    )
""")

for batch in batches:
    # Convert to SQL IN clause
    code_list = ','.join(f"'{code}'" for code in batch)

    # create query to insert data into new london_LSOAs_vol table
    query = f"""
        INSERT INTO london_LSOAs_vol
        SELECT
            l.LSOA21CD,
            l.Borough AS borough,
            COUNT(*) AS building_count,
            SUM(b.volume) AS total_volume,
            l.geom
        FROM london_lsoa l
        JOIN buildings b
        ON ST_Intersects(l.geom, ST_GeomFromText(b.geometry))
        WHERE l.LSOA21CD IN ({code_list})
        GROUP BY l.LSOA21CD, l.geom, l.Borough;
    """
    # execute the query created above
    con.execute(query)


# Calculate the building mass density
# Building Mass Density = LSOA area x total building volume

# Calculate area for each LSOA (assuming CRS in meters)
con.execute("""
    ALTER TABLE london_LSOAs_vol ADD COLUMN area_m2 DOUBLE;
    UPDATE london_LSOAs_vol
    SET area_m2 = ST_Area(geom);
""")

# Calculate building mass density for each LSOA
con.execute("""
    ALTER TABLE london_LSOAs_vol ADD COLUMN building_mass_density DOUBLE;
    UPDATE london_LSOAs_vol
    SET building_mass_density = total_volume / area_m2;
""")


## Tidy data into a clean table the geometry is removed
# so we can join back to the cleam lsoa geometry table later
con.execute("""
    DROP TABLE IF EXISTS lsoa_bmd;

    CREATE TABLE lsoa_bmd AS
    SELECT LSOA21CD, borough, total_volume, building_count, building_mass_density
    FROM london_LSOAs_vol;
""")

# Create a new joined table that includes geometries and attributes
con.execute("""
CREATE OR REPLACE TABLE london_lsoa_bmd AS
SELECT
    l.LSOA21CD,
    l.LSOA21NM,
    l.borough,
    b.total_volume,
    b.building_count,
    b.building_mass_density,
    l.geom
FROM london_lsoa l
JOIN lsoa_bmd b
ON l.LSOA21CD = b.LSOA21CD;
""")


# Prepare data for export by first loading
# london_lsoa_bmd from DuckDB into a DataFrame with WKT geometry

gdf_lsoa_bmd = con.execute("""
    SELECT *, ST_AsText(geom) AS wkt_geom  -- Convert geometry to WKT
    FROM london_lsoa_bmd
""").df()


# import wkt from shapely library
from shapely import wkt

# Convert WKT to geometry
gdf_lsoa_bmd['geometry'] = gdf_lsoa_bmd['wkt_geom'].apply(wkt.loads)

# Create GeoDataFrame
gdf_lsoa_bmd = gpd.GeoDataFrame(gdf_lsoa_bmd, geometry='geometry')

# Set CRS if needed (EPSG:27700 = British National Grid)
gdf_lsoa_bmd.set_crs("EPSG:27700", inplace=True)

# Rename columns to avoid truncation 
# previously used as exported to shapefile but retained for cleanliness 
gdf_lsoa_bmd = gdf_lsoa_bmd.rename(
    columns={
        'LSOA21CD': 'LSOA_CD',
        'Borough': 'BORO',
        'building_mass_density': 'bldg_dens',
        'building_count': 'bldg_cnt',
        'total_volume': 'tot_vol'
    }
)

# Keep only necessary columns
columns_to_keep = ['LSOA_CD', 'BORO', 'bldg_dens', 'bldg_cnt', 'tot_vol', 'wkt_geom', 'geometry']
gdf_lsoa_bmd = gpd.GeoDataFrame(gdf_lsoa_bmd[columns_to_keep], geometry='geometry')

# Export to GeoJSON
output_geojson_path = "data/geojson/lsoa_bmd.geojson"
os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)
gdf_lsoa_bmd.to_file(output_geojson_path, driver="GeoJSON")

print(f"GeoJSON exported to: {output_geojson_path}")

