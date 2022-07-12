import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

st.title('MSCI 436 Final Project')
st.header("Detecting Fradulent Transactions")

# @st.cache(allow_output_mutation=True)
def load_model():
    return pickle.load(open('model.pickle.dat', 'rb'))

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("data/fraudTest.csv", parse_dates=['trans_date_trans_time'])
    df['date'] = df['trans_date_trans_time'].dt.date
    df['weekday'] = df['trans_date_trans_time'].dt.day_name()
    df.rename(columns={"long": "lon"}, inplace=True)
    # return df[df.is_fraud == 1].sample(frac=0.5)
    return df[df.is_fraud == 1]


model = load_model()

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)

data = load_data()
st.write(data)

### -- Date input
# @st.experimental_memo
def get_time_series_chart(data):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none"
    )

    lines = (
        alt.Chart(data, title="Fradulent Transactions over Time")
        .mark_line()
        .encode(x="date", y="count()")
    )

    points = lines.transform_filter(hover).mark_circle(size=65)

    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(date)",
            y="count()",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip("count()", title="Count"),
            ],
        )
        .add_selection(hover)
    )

    return (lines+points+tooltips).interactive()


col1, col2 = st.columns(2)
min_start=min(data['date'])
max_end=max(data['date'])

with col1:
    start_date = st.date_input(
        label="Start Date",
        value=min_start,
        min_value=min_start,
        max_value=max_end,
    )

with col2:
    end_date = st.date_input(
        label="End Date",
        value=max_end,
        min_value=min_start,
        max_value=max_end,
    )

time_data = data[(data["date"] >= start_date) & (data["date"] <= end_date)]
time_series_chart = get_time_series_chart(time_data)

st.altair_chart(time_series_chart)


## End date input --

## -- Weekly date chart
# @st.experimental_memo
def get_weekly_chart(data):
    hover = alt.selection_single(
        fields=["weekday"],
        nearest=True,
        on="mouseover",
        empty="none"
    )

    lines = (
        alt.Chart(data, title="Weekly view of Fradulent Transactions")
        .mark_bar()
        .encode(x=alt.X("weekday", sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]), y="count()")
    )

    points = lines.transform_filter(hover).mark_circle(size=65)

    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x=alt.X("weekday", sort=["Monday", "Tuesday", "Wednesday", "Friday", "Saturday", "Sunday"]),
            y="count()",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("weekday", title="Day of Week"),
                alt.Tooltip("count()", title="Count"),
            ],
        )
        .add_selection(hover)
    )

    return (lines+points+tooltips).interactive()

st.altair_chart(get_weekly_chart(data))

## End weekly day chart --

# US MAP --
pos_data = data.filter(items=["lon", "lat", "state"])
pos_data.reset_index(drop=True, inplace=True)

# pos_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

st.pydeck_chart(
    pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=40,
            longitude=-81.5488,
            zoom=10,
            # max_zoom=16,
            # pitch=0,
            # bearing=0,
            # height=900,
            # width=None,
        ),
        layers=[
            pdk.Layer(
               'HexagonLayer',
                data=pos_data,
                get_position='[lon, lat]',
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                # pickable=True,
                # opacity=0.5,
                # stroked=True,
                # filled=True,
                # extruded=True,
                # wireframe=True,
                # # get_elevation=50,
                # elevation_scale=50,
                # get_fill_color=f"[R, G, B]",
                # # # get_fill_color=color_exp,
                # get_line_color=[0, 0, 0],
                # get_line_width=2,
                # line_width_min_pixels=1,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=pos_data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    )
)
st.write(pos_data)

# df = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

# st.pydeck_chart(pdk.Deck(
#      map_style='mapbox://styles/mapbox/light-v9',
#      initial_view_state=pdk.ViewState(
#          latitude=37.76,
#          longitude=-122.4,
#          zoom=11,
#          pitch=50,
#      ),
#      layers=[
#          pdk.Layer(
#             'HexagonLayer',
#             data=df,
#             get_position='[lon, lat]',
#             radius=200,
#             elevation_scale=4,
#             elevation_range=[0, 1000],
#             pickable=True,
#             extruded=True,
#          ),
#          pdk.Layer(
#              'ScatterplotLayer',
#              data=df,
#              get_position='[lon, lat]',
#              get_color='[200, 30, 0, 160]',
#              get_radius=200,
#          ),
#      ],
#  ))

## -- End US MAP