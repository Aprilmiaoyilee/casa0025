# this is a hello world file showing how to integrate Google Earth Engine with Streamlit
def mask_s2_clouds(image):
  """Masks clouds in a Sentinel-2 image using the QA band.

  Args:
      image (ee.Image): A Sentinel-2 image.

  Returns:
      ee.Image: A cloud-masked Sentinel-2 image.
  """
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )

  return image.updateMask(mask).divide(10000)


import geemap
import os

import ee
import streamlit as st
import geemap
import os
import numpy as np
import geemap.foliumap as geemap
import geopandas as gp
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
import json
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
import shapely
# these are the imports to make the website work
from google.oauth2 import service_account
from ee import oauth

def get_auth():
    service_account_keys = st.secrets["ee_keys"]
    credentials = service_account.Credentials.from_service_account_info(service_account_keys,
     scopes=oauth.SCOPES)
    # geemap.ee_initialize(credentials)
    ee.Initialize(credentials)
    return "Successfully authenticated with Google Earth Engine"

get_auth()

st.set_page_config(
    layout="wide"
)
st.title("Google Earth Engine - Demo application")

# os.environ["EARTHENGINE_TOKEN"] = st.secrets["google_earth_engine"]["EARTHENGINE_TOKEN"]
# token = st.secrets["google_earth_engine"]["refresh_token"]
# adding geemap.Map() as the first thing for the app

# geemap.ee_initialize()

# cache variables

# Initialize session state variables
if 'gdf_results' not in st.session_state:
    st.session_state.gdf_results = None

if 'time_series_gdf' not in st.session_state:
    st.session_state.time_series_gdf = None
if 'ndvi' not in st.session_state:
    st.session_state.ndvi = None
if 'ee_boroughs' not in st.session_state:
    st.session_state.ee_boroughs = None
if 'nox_layer' not in st.session_state:
    st.session_state.nox_layer = None
if 'london_midpoint_latitude' not in st.session_state:
    st.session_state.london_midpoint_latitude = None
if 'london_midpoint_longitude' not in st.session_state:
    st.session_state.london_midpoint_longitude = None
if "fc_results" not in st.session_state:
    st.session_state.fc_results = None
if "temperature_layer" not in st.session_state:
    st.session_state.temperature_layer = None
if "london_boroughs_over_65" not in st.session_state:
    st.session_state.london_boroughs_over_65 = None
if "london_boroughs_over_65_map" not in st.session_state:
    st.session_state.london_boroughs_over_65_map = None
if "london_boroughs_over_65_map_key" not in st.session_state:
    st.session_state.london_boroughs_over_65_map_key = None
if "lad_data" not in st.session_state:
    st.session_state.lad_data = None
if "london_lsoa_over_65_gdf" not in st.session_state:
    st.session_state.london_lsoa_over_65_gdf = None
if "gdf_boroughs" not in st.session_state:
    st.session_state.gdf_boroughs = None
if "gdf_results" not in st.session_state:
    st.session_state.gdf_results = None
if "buildings_data_gdf" not in st.session_state:
    st.session_state.buildings_data_gdf = None
if "buildings_data_df" not in st.session_state:
    st.session_state.buildings_data_df = None
    
    
col1_original, col2_original = st.columns([2,12])

with col1_original:
    collection = st.selectbox("Select satellite image collection", ["NAIP", "Landsat","Sentinel-2","NDVI London","Nitrogen","Temperature","Population","Index"])

    # os.environ["EARTHENGINE_TOKEN"] = st.secrets["google_earth_engine"]["refresh_token"]

    # Initialize Earth Engine - should authenticate automatically using credentials
    # Authenticate and initialize
    # try:
    #     ee.Initialize()
    # except Exception as e:
    #     ee.Authenticate()  # Trigger OAuth flow
    #     ee.Initialize()

    # Load credentials from Streamlit secrets
    service_account_info = st.secrets["google_earth_engine"]["project"]

    # Authenticate using the service account
    # ee.Initialize(project=service_account_info)

    # this is for the streamlit deployed version
    # geemap.ee_initialize(project=service_account_info)

    st.success("Successfully authenticated with Google Earth Engine")

with col2_original:
    if collection == "NAIP":
        # Define the map center and zoom level
        center = [40, -100]
        zoom = 4


        location_tuple = (-74.00, 40.71)
        loc = ee.Geometry.Point(-115.1482, 36.0831)
        loc = ee.Geometry.Point(location_tuple)
        # Define the centroid as a point geometry


        # Apply a buffer to define the ROI (in meters)
        roi = loc.buffer(500)  # Example: 5 km buffer

        Map = geemap.Map()
        dataset = ee.ImageCollection('USDA/NAIP/DOQQ').filterBounds(roi).filterDate(
            '2022-03-20', '2023-07-31'
        )#.map(roi)

        image = dataset.median().clip(roi)


        visualization = {
            #'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
            'bands': ['R','G','B'],

            'min': 0,
            'max': 255,
        }

        m = geemap.Map()
        m.set_center(location_tuple[0], location_tuple[1],16)
        m.add_layer(dataset, visualization, 'True Color (432)')
        m.to_streamlit(height=600)

    elif collection == "Landsat":
        dataset = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterDate(
        '2022-01-01', '2022-02-01'
        )


        # Applies scaling factors.
        def apply_scale_factors(image):
            optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
            thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
            return image.addBands(optical_bands, None, True).addBands(
                thermal_bands, None, True
            )


        dataset = dataset.map(apply_scale_factors)

        visualization = {
            'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
            'min': 0.0,
            'max': 0.3,
        }

        m = geemap.Map()
        m.set_center(0.12, 51.52, 10)
        m.add_layer(dataset.median(), visualization, 'True Color (432)')
        m.to_streamlit(height=600)

    elif collection == "Sentinel-2":
        dataset = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate('2020-01-01', '2020-01-30')
        # Pre-filter to get less cloudy granules.
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_s2_clouds)
        )

        visualization = {
            'min': 0.0,
            'max': 0.3,
            'bands': ['B4', 'B3', 'B2'],
        }

        m = geemap.Map()
        m.set_center(0.12, 51.52, 10)
        m.add_layer(dataset.median(), visualization, 'RGB')
        m.to_streamlit(height=600)

    elif collection == "NDVI London":

        with st.spinner("Loading London Boroughs..."):
            # 1️ Load the London borough boundaries 
            gdf_boroughs = gp.read_file(
                'data/london_lad.geojson'
            )#.head(5)
            gdf_boroughs.columns = [x.lower() for x in gdf_boroughs.columns]
            gdf_boroughs = gdf_boroughs[["lad11nm","geometry"]].rename(columns={"lad11nm":"borough_name"})
            # st.write(f"{gdf_boroughs.columns}")
            # [['lad22nm', 'geometry']]  # select only name + geometry

            # calculate the midpoint of london
            london_midpoint_latitude, london_midpoint_longitude = gdf_boroughs.to_crs(4326).geometry.centroid.y.mean(), gdf_boroughs.to_crs(4326).geometry.centroid.x.mean()
            # st.write(f"London midpoint: {london_midpoint_latitude}, {london_midpoint_longitude}")

            if st.session_state.gdf_results is None:


                # 2️ Convert GeoDataFrame → EE FeatureCollection
                ee_boroughs = geemap.geopandas_to_ee(gdf_boroughs, geodesic=False)
                st.session_state.ee_boroughs = ee_boroughs
                # 3️ Build Sentinel‑2 NDVI composite
                sentinel = (
                    ee.ImageCollection('COPERNICUS/S2_SR')
                    .filterBounds(ee_boroughs)
                    .filterDate('2020-06-01', '2020-09-30')
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                    .median()
                    .clip(ee_boroughs)
                )
                ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
                st.session_state.ndvi = ndvi

                # 4️ Sum NDVI per borough
                fc_results = ndvi.reduceRegions(
                    collection=ee_boroughs,
                    reducer=ee.Reducer.median(),
                    scale=10,
                    crs='EPSG:27700'
                )
                
                # 5️⃣Pull results client‑side as GeoJSON → GeoDataFrame
                geojson = fc_results.getInfo()
                
                gdf_results = gp.GeoDataFrame.from_features(geojson['features']).rename(columns={'NAME': 'MSOA Name',"median": "NDVI"})
                st.session_state.gdf_results = gdf_results
            else:
                gdf_results = st.session_state.gdf_results
                ndvi = st.session_state.ndvi
                ee_boroughs = st.session_state.ee_boroughs
            col1, col2 = st.columns([8,5])

            with col1:

                m = geemap.Map()
                m.set_center(london_midpoint_longitude, london_midpoint_latitude, 10)

                # 🖼️ Add the NDVI image itself
                ndvi_vis = {
                    'min': 0.0,
                    'max': 1.0,
                    'palette': ['red', 'yellow', 'green']
                }
                m.addLayer(ndvi, ndvi_vis, 'NDVI')

                # (Optional) Outline borough boundaries on top
                m.addLayer(ee_boroughs.style(**{
                    'color': '000000', 
                    'fillColor': '00000000', 
                    'width': 1
                }), {}, 'Borough boundaries')
                
                st.success("Successfully collected data from GEE")


                m.to_streamlit(height=600)

                

            with col2:

                # now calculate the average NDVI per borough
                gdf_results['NDVI'] = gdf_results['NDVI'].astype(float)
                gdf_results['area'] = gdf_results['geometry'].area
                # sum the NDVI value for each borough
                gdf_results_agg = gdf_results[["borough_name","NDVI","area"]].groupby('borough_name').mean().reset_index()
                #normalise the NDVI value by dividing it by the area
                gdf_results_agg['Average NDVI'] = gdf_results_agg['NDVI'] # / gdf_results_agg['area']
                gdf_results_agg = gdf_results_agg.sort_values("Average NDVI", ascending=True)
                # visualise using matplotlib
                import matplotlib.pyplot as plt

                st.write("Average NDVI per Borough - Top 10")
                fig = plt.figure(figsize=(6, 14))
                ax = plt.gca()

                # then set background to be clear
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)

                gdf_results_agg.set_index(['borough_name'])['Average NDVI'].plot(kind='barh', ax=ax)
                plt.xlabel('Borough')
                plt.ylabel('Average NDVI')
                # plt.title('Average NDVI per Borough')
                plt.xticks(rotation=0, fontsize=12)
                plt.show()
                sns.despine()
                st.pyplot(fig, use_container_width=True)
                
                #st.dataframe(gdf_results.drop(columns=['geometry']))

        # now we're going to calculate the time series results for this 
        with st.spinner("Calculating monthly NDVI..."):
            if st.session_state.time_series_gdf is None:
                # first we get today's date
                today = datetime.now().strftime("%Y-%m-%d")
                # then we're going to get the today's date a year ago
                one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

                # now we're going to get the monthly start and end dates for this period
                # we'll make a loop and generate to lists the start dates and the end dates
                start_dates_list = []
                end_dates_list = []

                for i in range(1,13):
                    period_start_date_increment = int(one_year_ago.split("-")[1])
                    period_start_date_increment = period_start_date_increment + (i-1)
                    if len(str(period_start_date_increment)) == 1:
                        period_start_date_increment = "0"+str(period_start_date_increment)
                    else:
                        period_start_date_increment = str(period_start_date_increment)

                    # in the case that the period start date incremement is greater than 12 we'll update the year and the month value
                    if int(period_start_date_increment) > 12 :
                        period_start_date_increment = end_dates_list[i-2].split("-")[1]
                        year_value = end_dates_list[i-2].split("-")[0]
                        period_start_date = year_value+"-"+   period_start_date_increment    +"-"+ one_year_ago.split("-")[-1]
                    else:
                        period_start_date = one_year_ago.split("-")[0]+"-"+   period_start_date_increment    +"-"+ one_year_ago.split("-")[-1]
                    print(period_start_date_increment)

                    interim_period_end_date = ((datetime.now() - timedelta(days=365)) + timedelta(i * 30)).strftime("%Y-%m-%d")
                    if period_start_date.split("-")[-1] != interim_period_end_date.split("-")[-1]:
                        # calculate the difference in days
                        diff_in_days = int(period_start_date.split("-")[-1]) - int(interim_period_end_date.split("-")[-1])
                        # print(diff_in_days) 
                        period_end_date = ((datetime.now() - timedelta(days=365)) + timedelta((i * 30)+diff_in_days)).strftime("%Y-%m-%d")
                    else:
                        period_end_date = ((datetime.now() - timedelta(days=365)) + timedelta(i * 30)).strftime("%Y-%m-%d")
                    start_dates_list.append(period_start_date)
                    end_dates_list.append(period_end_date)
                    print(period_start_date,"|",period_end_date)


                time_series_gdf_container = []
                # now that we have the date ranges we care about we're going to iterate and collect the values for each Borough and time period
                for month_value in range(len(end_dates_list)):
                    # st.write(start_dates_list[month_value], end_dates_list[month_value])
                    sentinel = (
                        ee.ImageCollection('COPERNICUS/S2_SR')
                        .filterBounds(ee_boroughs)
                        .filterDate(start_dates_list[month_value], end_dates_list[month_value])
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                        .median()
                        .clip(ee_boroughs)
                    )
                    ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
                    print(ndvi)

                    # 4️ Sum NDVI per borough
                    fc_results = ndvi.reduceRegions(
                        collection=ee_boroughs,
                        reducer=ee.Reducer.mean(),
                        scale=10,
                        crs='EPSG:27700'
                    )
                    st.session_state.fc_results = fc_results
                    # 5️⃣Pull results client‑side as GeoJSON → GeoDataFrame
                    geojson = fc_results.getInfo()
                    temp_gdf_results = gp.GeoDataFrame.from_features(geojson['features']).rename(columns={'NAME': 'borough_name',"mean": "NDVI"})
                    temp_gdf_results["start_date"] = pd.to_datetime(start_dates_list[month_value])
                    temp_gdf_results["end_date"] = end_dates_list[month_value]
                    time_series_gdf_container.append(temp_gdf_results)

                
                    time_series_gdf = pd.concat(time_series_gdf_container, ignore_index=True).drop(columns=['geometry'])
                    st.session_state.time_series_gdf = time_series_gdf
            else:
                time_series_gdf = st.session_state.time_series_gdf
                fc_results = st.session_state.fc_results
            # st.dataframe(time_series_gdf)
            borough_names = time_series_gdf['borough_name'].value_counts().index.tolist()
            # st.write(borough_names)
            borough_names.sort()
            # we're going to have a select box to select the borough name
            borough_name_select = st.selectbox("Select a borough", ["All"]+borough_names+["None"])
            # set the default to be all
            if borough_name_select != "None":
                if borough_name_select == "All":
                    # now we're going to filter the time series gdf to only include the selected borough
                    viz_time_series_gdf = time_series_gdf.copy()
                    viz_time_series_gdf["borough_name"] = "All"
                    viz_time_series_gdf = viz_time_series_gdf[["start_date","NDVI","borough_name"]].groupby(['start_date','borough_name']).mean().reset_index()
                    # now make a line chart of the time series using matplotlib
                    import matplotlib.pyplot as plt

                    st.write("Time Series of NDVI")
                    fig = plt.figure(figsize=(15, 5))
                    ax = plt.gca()
                    sns.lineplot(x='start_date', y='NDVI',hue='borough_name', data=viz_time_series_gdf)
                    plt.xticks(rotation=0, fontsize=12)
                    plt.show()
                    sns.despine()
                    st.pyplot(fig, use_container_width=True)
                else:
                    # now we're going to filter the time series gdf to only include the selected borough
                    viz_time_series_gdf = time_series_gdf[time_series_gdf['borough_name'] == borough_name_select]

                    # now make a line chart of the time series using matplotlib
                    import matplotlib.pyplot as plt

                    st.write("Time Series of NDVI")
                    fig = plt.figure(figsize=(15, 5))
                    ax = plt.gca()
                    sns.lineplot(x='start_date', y='NDVI',hue='borough_name', data=viz_time_series_gdf)
                    plt.xticks(rotation=0, fontsize=12)
                    plt.show()
                    sns.despine()
                    st.pyplot(fig, use_container_width=True)
                
                    
    elif collection == "Nitrogen":
        with st.spinner("Loading London Boroughs..."):
        # 2️ Convert GeoDataFrame → EE FeatureCollection

            if st.session_state.ee_boroughs is None:
                gdf_boroughs = gp.read_file(
                    'data/london_lad.geojson'
                )#.head(5)
                gdf_boroughs.columns = [x.lower() for x in gdf_boroughs.columns]
                gdf_boroughs = gdf_boroughs[["lad11nm","geometry"]].rename(columns={"lad11nm":"borough_name"})
                ee_boroughs = geemap.geopandas_to_ee(gdf_boroughs, geodesic=False)

                # calculate the midpoint of london
                london_midpoint_latitude, london_midpoint_longitude = gdf_boroughs.to_crs(4326).geometry.centroid.y.mean(), gdf_boroughs.to_crs(4326).geometry.centroid.x.mean()
                st.session_state.ee_boroughs = ee_boroughs
                st.session_state.london_midpoint_latitude = london_midpoint_latitude
                st.session_state.london_midpoint_longitude = london_midpoint_longitude
            else:
                ee_boroughs = st.session_state.ee_boroughs
                london_midpoint_latitude = st.session_state.london_midpoint_latitude
                london_midpoint_longitude = st.session_state.london_midpoint_longitude

           
            # now we're going to load the nitrogen data from Sentinel-5P
            nox_layer = (ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
                        .filterBounds(ee_boroughs)

                        .select('tropospheric_NO2_column_number_density')                    
                        .filterDate('2020-01-01', '2020-12-30')
                        # .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                        .median()
                        .clip(ee_boroughs)
                    )
            st.session_state.nox_layer = nox_layer
            
            
            visualization = {
                        'min': 0,
                        'max': 0.0002,
                        'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red'],
                        # "palette":['#fde725', '#5ec962','#21918c','#3b528b','#440154']
                        };
            m = geemap.Map()
            m.add_basemap("CartoDB.Positron")
            m.set_center(london_midpoint_longitude, london_midpoint_latitude, 10)
            m.add_layer(nox_layer, visualization, 'Nitrogen oxide')
            m.add_colorbar(
                    visualization,
                    label="Nitrogen Oxide",
                    layer_name="NOx",
                    orientation="vertical",
                    transparent_bg=True,
                
                )

            st.success("Successfully loaded nitrogen data")
            m.to_streamlit(height=600)


    elif collection == "Temperature":
        with st.spinner("Loading London Boroughs..."):
            # 2️ Convert GeoDataFrame → EE FeatureCollection

            if st.session_state.ee_boroughs is None:
                gdf_boroughs = gp.read_file(
                    'data/london_lad.geojson'
                )#.head(5)
                gdf_boroughs.columns = [x.lower() for x in gdf_boroughs.columns]
                gdf_boroughs = gdf_boroughs[["lad11nm","geometry"]].rename(columns={"lad11nm":"borough_name"})
                ee_boroughs = geemap.geopandas_to_ee(gdf_boroughs, geodesic=False)

                # calculate the midpoint of london
                london_midpoint_latitude, london_midpoint_longitude = gdf_boroughs.to_crs(4326).geometry.centroid.y.mean(), gdf_boroughs.to_crs(4326).geometry.centroid.x.mean()
                st.session_state.ee_boroughs = ee_boroughs
                st.session_state.london_midpoint_latitude = london_midpoint_latitude
                st.session_state.london_midpoint_longitude = london_midpoint_longitude
            else:
                ee_boroughs = st.session_state.ee_boroughs
                london_midpoint_latitude = st.session_state.london_midpoint_latitude
                london_midpoint_longitude = st.session_state.london_midpoint_longitude


            # Applies scaling factors.
            def apply_scale_factors(image):
                optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
                thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
                return image.addBands(optical_bands, None, True).addBands(
                    thermal_bands, None, True
                )
            # now we're going to load the temperature data from Sentinel-5P


            dataset = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterDate(
                        '2022-05-01', '2022-10-31'
                    )
            
            dataset = dataset.map(apply_scale_factors)
            
            temperature_layer = dataset.select('ST_B10').median().clip(ee_boroughs)

            st.session_state.temperature_layer = temperature_layer

            # 4️ Sum NDVI per borough
            temperature_results = temperature_layer.reduceRegions(
                collection=ee_boroughs,
                reducer=ee.Reducer.median(),
                scale=10,
                crs='EPSG:27700'
            )
                       
            # 5️⃣Pull results client‑side as GeoJSON → GeoDataFrame
            geojson = temperature_results.getInfo()
            
            temperature_gdf_results = gp.GeoDataFrame.from_features(geojson['features']).rename(columns={'NAME': 'MSOA Name',"median": "surface_temperature"})
            # st.session_state.gdf_results = gdf_results
                        # convert the temperature to a geodataframe
            # st.dataframe(temperature_gdf_results)


            # st.write(temperature_layer)
            visualization = {
                'min': temperature_gdf_results["surface_temperature"].min(),
                'max': temperature_gdf_results["surface_temperature"].max(),
                'palette': ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#8c2d04'],
            }


            # now we're going to add in the vector data for the london boroughs for number of people over 65 from a geojson file
            # london_lsoa_over_65_gdf = pd.read_parquet('data/london_percentage_of_population_over_65.parquet.gzip')
            # # convert the wkt geometry to a shapely geometry
            # london_lsoa_over_65_gdf["geometry"] = london_lsoa_over_65_gdf["geometry"].apply(shapely.wkt.loads)
            # # convert this to a geodataframe
            # london_lsoa_over_65_gdf = gp.GeoDataFrame(london_lsoa_over_65_gdf, geometry="geometry", crs=4326)
            
            # with open('data/london_percentage_of_population_over_65.geojson') as f:
            #      london_boroughs_over_65_geojson = json.load(f)
            # # maybe I can convert the geodatafraame to a geojson and then to an df
            # london_boroughs_over_65_df = geemap.geojson_to_df(london_boroughs_over_65_geojson).head(10)
            # london_boroughs_over_65_df["total_pop_over_65_years_old"] = london_boroughs_over_65_df["total_pop_over_65_years_old"].astype(int)
            # st.write(type(london_boroughs_over_65_df))

            # ee_fc = geemap.gdf_to_ee(london_boroughs_over_65)  # Requires geemap >= 0.29.0 [3]

            # choropleth_style = {
            #                     'fillColor': {
            #                         'property': 'total_pop_over_65_years_old',  # Target column
            #                         'palette': ['blue', 'limegreen', 'red'],  # Color gradient
            #                     },
            #                     'color': 'black',
            #                     # 'fillOpacity': 0.7,
            #                     'width': 1
            #                 }

            # styled_layer = ee_fc.style(**choropleth_style)  # Apply styling [3]

            
            # london_boroughs_over_65.columns = [x.lower() for x in london_boroughs_over_65.columns]
            # london_boroughs_over_65 = london_boroughs_over_65[["geography","total_pop_over_65_years_old","geometry"]]
            # london_boroughs_over_65_ee = geemap.geopandas_to_ee(london_boroughs_over_65, geodesic=False)
            # st.markdown(london_boroughs_over_65_ee)

            # now we're going to add this to the map
            # st.dataframe(london_boroughs_over_65_df)


            
            


            m = geemap.Map()
            m.add_basemap("CartoDB.Positron")
            m.set_center(london_midpoint_longitude, london_midpoint_latitude, 10)
            m.add_layer(temperature_layer, visualization, 'Surface temperature')

            # we're going to add the london boroughs over 65 to the map
            # m.add_data(
            #     data=london_boroughs_over_65_df,
            #     column='total_pop_over_65_years_old',
            #     cmap='Blues',
            #     scheme='Quantiles',
            #     # legend_title='Population over 65',
            # )
            # m.add_gdf(
            #     london_boroughs_over_65,
            #     layer_name='London Boroughs over 65',
            #     column='total_pop_over_65_years_old',  # Column to style by
            #     cmap='YlOrRd',                         # Color map
            #     scheme='Quantiles',                    # Classification scheme
            #     legend_title='Population over 65',     # Legend title
            # )

            # m.add_colorbar(
            #         visualization,
            #         label="Surface temperature",
            #         layer_name="Surface temperature",
            #         orientation="vertical",
            #         transparent_bg=True,
                
            #     )
            
            st.success("Successfully loaded nitrogen data")
            m.to_streamlit(height=600)           

            # this is a test commit to the main branch
            

    elif collection == "Population":

        with st.spinner("Loading London Boroughs..."):
            # 2️ Convert GeoDataFrame → EE FeatureCollection
            if st.session_state.london_boroughs_over_65 is None:
                # now we're going to add in the vector data for the london boroughs for number of people over 65 from a geojson file
                # london_boroughs_over_65 = gp.read_file('data/london_percentage_of_population_over_65.geojson')#.head(10).to_crs(4326)
                london_boroughs_over_65 = pd.read_parquet('data/london_percentage_of_population_over_65.parquet.gzip')
                # convert the wkt geometry to a shapely geometry
                london_boroughs_over_65["geometry"] = london_boroughs_over_65["geometry"].apply(shapely.wkt.loads)
                # convert this to a geodataframe
                london_boroughs_over_65 = gp.GeoDataFrame(london_boroughs_over_65, geometry="geometry", crs=4326)


                # add this to session state
                st.session_state.london_boroughs_over_65 = london_boroughs_over_65
            else:
                london_boroughs_over_65 = st.session_state.london_boroughs_over_65
            
            # calculate the midpoint of london
            london_midpoint_latitude, london_midpoint_longitude = london_boroughs_over_65.to_crs(4326).geometry.centroid.y.mean(), london_boroughs_over_65.to_crs(4326).geometry.centroid.x.mean()
            
            if st.session_state.london_boroughs_over_65_map is None:
                # we'll plot this on a folium map
                m = london_boroughs_over_65.explore("total_pop_over_65_years_old", tiles="CartoDB.Positron", cmap="Blues", scheme="Quantiles", legend_title="Population over 65", style_kwds={'weight': 1})
            
                # we're going to cache this map as well 
                st.session_state.london_boroughs_over_65_map = m
            else:
                m = st.session_state.london_boroughs_over_65_map

            st.success("Successfully loaded population data")
            # st_folium(m, width=725)
            st_folium(m, width=725, returned_objects=[])
            # m.to_streamlit(height=600)


    elif collection == "Index":
        with st.spinner("Loading the index data..."):

            if st.session_state.lad_data is None:
            # so the first thing we're going to do is load the lad data
                gdf_boroughs = gp.read_file(
                    'data/london_lad.geojson'
                )#.head(5)
                # alternatively we can use the lsoa file
                gdf_boroughs.columns = [x.lower() for x in gdf_boroughs.columns]
                gdf_boroughs = gdf_boroughs[["lad11nm","geometry"]].rename(columns={"lad11nm":"borough_name"})
                st.session_state.gdf_boroughs = gdf_boroughs
            else:
                gdf_boroughs = st.session_state.gdf_boroughs


            # next we're going to load the population age data 

            if st.session_state.london_lsoa_over_65_gdf is None:
                # now we're going to add in the vector data for the london boroughs for number of people over 65 from a geojson file
                # london_lsoa_over_65_gdf = gp.read_file('data/london_percentage_of_population_over_65.geojson')#.head(10).to_crs(4326)
                # alternatively this can be loaded from a parquet file
                london_lsoa_over_65_gdf = pd.read_parquet('data/london_percentage_of_population_over_65.parquet.gzip')
                # convert the wkt geometry to a shapely geometry
                london_lsoa_over_65_gdf["geometry"] = london_lsoa_over_65_gdf["geometry"].apply(shapely.wkt.loads)
                # convert this to a geodataframe
                london_lsoa_over_65_gdf = gp.GeoDataFrame(london_lsoa_over_65_gdf, geometry="geometry", crs=4326)
                
                # add this to session state
                st.session_state.london_lsoa_over_65_gdf = london_lsoa_over_65_gdf
            else:
                london_lsoa_over_65_gdf = st.session_state.london_lsoa_over_65_gdf

            # calculate the midpoint of london
            london_midpoint_latitude, london_midpoint_longitude = london_lsoa_over_65_gdf.geometry.centroid.y.mean(), london_lsoa_over_65_gdf.geometry.centroid.x.mean()

            # next we're going to load in some buildings data
            if st.session_state.buildings_data_gdf is None:
                # buildings_data_gdf = gp.read_file('data/greater_london_buildings.geojson')

                # this is being changed to be an iterative load of parquet files
                buildings_container = []
                for i in range(11):
                    int_df = pd.read_parquet(f"data/chunked_greater_london_buildings_chunk_{i+1}.parquet.gzip")
                    buildings_container.append(int_df)
                buildings_data_gdf = pd.concat(buildings_container)
                # use shapely to the wkt geometry to an actual shapely geometry
                buildings_data_gdf["geometry"] = buildings_data_gdf["geometry"].apply(shapely.wkt.loads)
                # convert this to a geodataframe
                buildings_data_gdf = gp.GeoDataFrame(buildings_data_gdf, geometry="geometry", crs=4326)
                st.session_state.buildings_data_gdf = buildings_data_gdf
            else:
                buildings_data_gdf = st.session_state.buildings_data_gdf

            # ------------------------------------------------------------
            # now we're going to use the lad data to get the NDVI
            if st.session_state.gdf_results is None:


                # 2️ Convert GeoDataFrame → EE FeatureCollection
                ee_boroughs = geemap.geopandas_to_ee(gdf_boroughs, geodesic=False)
                st.session_state.ee_boroughs = ee_boroughs
                # 3️ Build Sentinel‑2 NDVI composite
                sentinel = (
                    ee.ImageCollection('COPERNICUS/S2_SR')
                    .filterBounds(ee_boroughs)
                    .filterDate('2020-06-01', '2020-09-30')
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                    .median()
                    .clip(ee_boroughs)
                )
                ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
                st.session_state.ndvi = ndvi

                # 4️ Sum NDVI per borough
                fc_results = ndvi.reduceRegions(
                    collection=ee_boroughs,
                    reducer=ee.Reducer.median(),
                    scale=10,
                    crs='EPSG:27700'
                )
                
                # 5️⃣Pull results client‑side as GeoJSON → GeoDataFrame
                geojson = fc_results.getInfo()
                
                gdf_results = gp.GeoDataFrame.from_features(geojson['features']).rename(columns={'NAME': 'MSOA Name',"median": "NDVI"})
                # for ndvi we will need to invert these values 
                gdf_results["NDVI"] = 1 / gdf_results["NDVI"]

                st.session_state.gdf_results = gdf_results
            else:
                gdf_results = st.session_state.gdf_results
                ndvi = st.session_state.ndvi
                ee_boroughs = st.session_state.ee_boroughs

            # ------------------------------------------------------------
            # now we're going to get the temperature data

            # Applies scaling factors.
            def apply_scale_factors(image):
                optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
                thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
                return image.addBands(optical_bands, None, True).addBands(
                    thermal_bands, None, True
                )
            # now we're going to load the temperature data from Sentinel-5P


            dataset = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterDate(
                        '2022-05-01', '2022-10-31'
                    )
            
            dataset = dataset.map(apply_scale_factors)
            
            temperature_layer = dataset.select('ST_B10').median().clip(ee_boroughs)

            st.session_state.temperature_layer = temperature_layer

            # 4️ Sum NDVI per borough
            temperature_results = temperature_layer.reduceRegions(
                collection=ee_boroughs,
                reducer=ee.Reducer.median(),
                scale=10,
                crs='EPSG:27700'
            )
                       
            # 5️⃣Pull results client‑side as GeoJSON → GeoDataFrame
            geojson = temperature_results.getInfo()
            
            temperature_gdf_results = gp.GeoDataFrame.from_features(geojson['features']).rename(columns={'NAME': 'MSOA Name',"median": "surface_temperature"})
            # ------------------------------------------------------------
            # next we will aggregate the age data to the borough level
            # do a spatial join between the age data and the lad data 

            london_boroughs_over_65 = london_lsoa_over_65_gdf.to_crs(4326).sjoin(gdf_boroughs.to_crs(4326), how="left")
            london_boroughs_over_65_gdf = london_boroughs_over_65[["borough_name","total_pop_over_65_years_old"]].groupby("borough_name").sum().reset_index()

            # now we'll add this to the session state
            st.session_state.london_boroughs_over_65 = london_boroughs_over_65_gdf

            # ------------------------------------------------------------
            # we're going to get the number of buildings in each borough
            buildings_data_gdf["building_area"] = buildings_data_gdf.to_crs(27700).geometry.area
            buildings_data_df = buildings_data_gdf.to_crs(4326).sjoin(gdf_boroughs.to_crs(4326), how="left")
            # st.write(buildings_data_df.columns)
            # buildings_data_df = buildings_data_df[["borough_name","osm_id"]].groupby("borough_name").nunique().reset_index().rename(columns={"osm_id":"number_of_buildings"})
            buildings_data_df = buildings_data_df[["borough_name","building_area"]].groupby("borough_name").sum().reset_index()
            # make a column where we sum the total area of the boroughs
            buildings_data_df_area = gdf_boroughs.copy()
            buildings_data_df_area["area"] = buildings_data_df_area.to_crs(27700).geometry.area
            # merge buildings_data_df with buildings_data_df_area
            buildings_data_df = buildings_data_df.merge(buildings_data_df_area[["borough_name","area"]], on="borough_name", how="left")
            buildings_data_df["building_density"] = buildings_data_df["building_area"] / buildings_data_df["area"]
            buildings_data_df = buildings_data_df.drop(columns=["area","building_area"])
            st.session_state.buildings_data_df = buildings_data_df

            # so now we'll check the dataframes we've got so far
            st.write("Age dataframe")
            st.dataframe(london_boroughs_over_65_gdf)
            st.write("LAD dataframe")
            st.dataframe(gdf_boroughs)
            st.write("NDVI dataframe")
            st.dataframe(gdf_results)
            st.write("Temperature dataframe")
            st.dataframe(temperature_gdf_results)
            st.write("Buildings dataframe")
            st.dataframe(buildings_data_df)


            # now we are going to merge these altogether
            raw_index_values_gdf_boroughs = gdf_boroughs.merge(london_boroughs_over_65_gdf, on="borough_name", how="left").merge(temperature_gdf_results.drop(columns=["geometry"]), on="borough_name", how="left").merge(gdf_results.drop(columns=["geometry"]), on="borough_name", how="left").merge(buildings_data_df, on="borough_name", how="left")

            # st.write("Merged dataframe")
            # st.dataframe(raw_index_values_gdf_boroughs)


            # okay so now we're going to normalise the data values  using sklearn min max scaler
            scaler = MinMaxScaler()
            # first we'll get the columns we want to normalise
            columns_to_normalise = ["NDVI","surface_temperature","total_pop_over_65_years_old","building_density"]

            for column in columns_to_normalise:
                raw_index_values_gdf_boroughs[f"{column}_normalised"] = scaler.fit_transform(raw_index_values_gdf_boroughs[[column]])

            st.write("Normalised dataframe")
            st.dataframe(raw_index_values_gdf_boroughs)


            # ------------------------------------------------------------
            normalised_columns = [x for x in raw_index_values_gdf_boroughs.columns if "normalised" in x]
            for column in normalised_columns:
                # now we're going to weight this by 25% for each of the normalised values
                raw_index_values_gdf_boroughs[f"{column}_weighted"] = raw_index_values_gdf_boroughs[column] * 0.25


            st.write("Weighted dataframe")
            weighted_df = raw_index_values_gdf_boroughs[["borough_name"]+[x for x in raw_index_values_gdf_boroughs.columns if "weighted" in x] + ["geometry"]]
            st.dataframe(weighted_df)

            # ------------------------------------------------------------
            # now lastly we're going to sum these up to get the final index values
            weighted_df["index_value"] = weighted_df[[x for x in weighted_df.columns if "weighted" in x]].sum(axis=1)
            weighted_columns = [x for x in weighted_df.columns if "weighted" in x]

            st.write("Final index dataframe")
            st.dataframe(weighted_df)

            # ------------------------------------------------------------

            viz_layer = st.selectbox("Select the layer you'd like to show", ["index_value"] + weighted_columns)
            if "NDVI" in viz_layer:
                weighted_df[viz_layer] = 1 / (weighted_df[viz_layer] + 0.01)
            # now we're going to add this to the map
            m = weighted_df.explore(viz_layer, tiles="CartoDB.Positron", cmap="Oranges", scheme="naturalbreaks", legend_title=viz_layer)
            # add a layer for each of the weighted columns
            # for column in weighted_columns:
            #     weighted_df.explore(column, tiles="CartoDB.Positron", cmap="Oranges", scheme="naturalbreaks", legend_title=column, m=m)
            st.success("Successfully loaded index data")
            st_folium(m, width=725, returned_objects=[])
            
            
            
            
            
            
            

