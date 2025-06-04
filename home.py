import geopandas as gpd
import pandas as pd
import pydeck as pdk
import shapely
import streamlit as st

from joblib import load

from california.config import (
    GEO_MEDIAN_DF_FILE,
    INTERIM_DATA_FILE,
    BEST_MODEL_FILE,
)

from california.streamlit_helpers import find_me_buttons, text_from_markdown

PAGE_TEXT_FILE = "./pages/01_home.md"


@st.cache_data
def load_data():
    return pd.read_parquet(INTERIM_DATA_FILE)


@st.cache_data
def load_geo_data():
    gdf_geo = gpd.read_parquet(GEO_MEDIAN_DF_FILE)

    # Explode MultiPolygons into individual polygons
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Function to check and fix invalid geometries
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)  # Fix invalid geometry
        # Orient the polygon to be counter-clockwise if it's a Polygon or MultiPolygon
        if isinstance(
            geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    # Apply the fix and orientation function to geometries
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fix_and_orient_geometry)

    # Extract polygon coordinates
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    # Apply the coordinate conversion and store in a new column
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(get_polygon_coordinates)

    return gdf_geo


@st.cache_resource
def load_model():
    return load(BEST_MODEL_FILE)


df = load_data()
gdf_geo = load_geo_data()
model = load_model()


st.title("Predict house prices in California")

content = text_from_markdown(PAGE_TEXT_FILE)

st.markdown("".join(content[0]))

counties = sorted(gdf_geo["name"].unique())

column1, column2 = st.columns(2)

with column1:
    with st.form(key="formulario"):
        select_county = st.selectbox("County", counties)

        longitude = gdf_geo.query("name == @select_county")["longitude"].values
        latitude = gdf_geo.query("name == @select_county")["latitude"].values

        housing_median_age = st.number_input(
            "House age", value=10, min_value=1, max_value=50
        )

        total_rooms = gdf_geo.query("name == @select_county")[
            "total_rooms"
        ].values
        population = gdf_geo.query("name == @select_county")[
            "population"
        ].values

        median_income = st.slider(
            "Median income (thousands US$)", 5.0, 100.0, 45.0, 5.0
        )

        median_income_scale = median_income / 10

        ocean_proximity = gdf_geo.query("name == @select_county")[
            "ocean_proximity"
        ].values

        median_income_cat = gdf_geo.query("name == @select_county")[
            "median_income_cat"
        ].values

        rooms_per_household = gdf_geo.query("name == @select_county")[
            "rooms_per_household"
        ].values
        bedrooms_per_room = gdf_geo.query("name == @select_county")[
            "bedrooms_per_room"
        ].values
        population_per_household = gdf_geo.query("name == @select_county")[
            "population_per_household"
        ].values

        model_input = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "population": population,
            "median_income": median_income_scale,
            "rooms_per_household": rooms_per_household,
            "population_per_household": population_per_household,
            "ocean_proximity": ocean_proximity,
            "median_income_cat": median_income_cat,
            "bedrooms_per_room": bedrooms_per_room,
        }

        df_model_input = pd.DataFrame(model_input)

        predict_button = st.form_submit_button("Predict price")

    if predict_button:
        price = model.predict(df_model_input)
        st.metric(label="Predict price: (US$)", value=f"{price[0]:.2f}")

with column2:
    view_state = pdk.ViewState(
        latitude=float(latitude[0]),
        longitude=float(longitude[0]),
        zoom=5,
        min_zoom=5,
        max_zoom=15,
    )

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf_geo[["name", "geometry"]],
        get_polygon="geometry",
        get_fill_color=[0, 0, 255, 100],
        get_line_color=[255, 255, 255],
        get_line_width=50,
        pickable=True,
        auto_highlight=True,
    )

    selected_county = gdf_geo.query("name == @select_county")

    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data=selected_county[["name", "geometry"]],
        get_polygon="geometry",
        get_fill_color=[255, 0, 0, 100],
        get_line_color=[0, 0, 0],
        get_line_width=500,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": "<b>County:</b> {name}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "fontsize": "10px",
        },
    }

    map = pdk.Deck(
        initial_view_state=view_state,
        map_style="light",
        layers=[polygon_layer, highlight_layer],
        tooltip=tooltip,
    )

    st.pydeck_chart(map)

columns = st.columns([1, 1, 1.2, 1, 1])

sites = ("linkedin", "portfolio", "github", "github_sponsors")
links = (
    "flsbustamante",
    "https://franciscobustamante.com.br",
    "chicolucio",
    "chicolucio",
)

with columns[2]:
    st.write("Developed by: Francisco Bustamante")
    for site, link in zip(sites, links):
        find_me_buttons(site, link)
