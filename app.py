import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from datetime import date

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
st.subheader("Time Visualizations")
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

st.altair_chart(time_series_chart, True)


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
            ]
        )
        .add_selection(hover)
    )

    return (lines+points+tooltips).interactive()

st.altair_chart(get_weekly_chart(data), True)

## End weekly day chart --

# US MAP --
st.subheader("Location Visualizations")
pos_data = data.filter(items=["lon", "lat", "state", "city"])
pos_data["city"] = pos_data["city"] + ", " + pos_data["state"]
st.pydeck_chart(
    pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=40,
            longitude=-95,
            zoom=3,
        ),
        layers=[
            pdk.Layer(
               'HexagonLayer',
                data=pos_data,
                get_position='[lon, lat]',
                radius=15000,
                elevation_scale=400,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                'Heatmaplayer',
                data=pos_data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=15000,
            ),
        ],
    )
)
state_data = pos_data.groupby('state').agg(count=('state', 'count')).sort_values(['count'], ascending=False).rename(columns={"count": "Number of Predicted Fradulent Transactions"}).reset_index()
st.table(state_data.head(10))

city_data = pos_data.groupby('city').agg(count=('city', 'count')).sort_values(['count'], ascending=False).rename(columns={"count": "Number of Predicted Fradulent Transactions"}).reset_index()
st.table(city_data.head(10))

## -- End US MAP

## Gender
st.subheader("Demographic Visualizations")

male_count=len(data[data['gender'] == 'M'])
female_count=len(data[data['gender'] == 'F'])
col1, col2 = st.columns(2)
col1.metric("Female", female_count)
col2.metric("Male", male_count)


def get_gender_chart(data):
    hover = alt.selection_single(
        fields=["gender"],
        nearest=True,
        on="mouseover",
        empty="none"
    )

    lines = (
        alt.Chart(data, title="Predicted Fradulent Transactions by Gender")
        .mark_bar()
        .encode(x="gender", y="count()", color=alt.Color("gender", legend=None))
    )

    points = lines.transform_filter(hover).mark_circle(size=65)

    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="gender",
            y="count()",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("gender", title="Gender"),
                alt.Tooltip("count()", title="Count"),
            ]
        )
        .add_selection(hover)
    )

    return (lines+points+tooltips).interactive()

st.altair_chart(get_gender_chart(data), True)

## -- End Gender

## Job
st.set_option('deprecation.showPyplotGlobalUse', False)
data[["job_a", "job_b"]] = data["job"].str.split(",", n=0, expand=True)
job_frequencies = dict(data.groupby("job_a")["job_a"].count())
wc = WordCloud(background_color="white", colormap="cividis")
wc.generate_from_frequencies(job_frequencies)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
st.pyplot()

def get_job_chart(data):
    hover = alt.selection_single(
        fields=["job_a", "count"],
        nearest=True,
        on="mouseover",
        empty="none"
    )

    lines = (
        alt.Chart(data, title="Predicted Fradulent Transactions by Occupation")
        .mark_bar()
        .encode(x="job_a", y="count", color=alt.Color("job_a", legend=None))
    )

    points = lines.transform_filter(hover).mark_circle(size=65)

    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="job_a",
            y="count",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("job_a", title="Occupation"),
                alt.Tooltip("count", title="Count"),
            ]
        )
        .add_selection(hover)
    )

    return (lines+points+tooltips).interactive()


st.altair_chart(get_job_chart(data.groupby("job_a").agg(count=("job_a", "count")).sort_values("count", ascending=False).head(10).reset_index()), True)

## -- End job

## Age ranges
def get_age_chart(data):
    hover = alt.selection_single(
        fields=["age"],
        nearest=True,
        on="mouseover",
        empty="none"
    )

    lines = (
        alt.Chart(data, title="Predicted Fradulent Transactions by Age")
        .mark_bar()
        .encode(x=alt.X("age", bin=True), y="count()", color=alt.Color("age", legend=None))
    )

    points = lines.transform_filter(hover).mark_circle(size=65)

    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="age",
            y="count()",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("age", title="Age"),
                alt.Tooltip("count()", title="Count"),
            ]
        )
        .add_selection(hover)
    )

    return (lines+points+tooltips).interactive()

def calculate_age(born):
    born = datetime.strptime(born, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

data["age"] = data["dob"].apply(calculate_age)
st.write(data)
st.altair_chart(get_age_chart(data), True)

## -- End age ranges

## TX ranges
## Age ranges
def get_tx_chart(data):
    return (
        alt.Chart(data, title="Predicted Fradulent Transactions by Transaction Amount")
        .mark_bar()
        .encode(x=alt.X("amt", bin=True), y="count()", color=alt.Color("amt", legend=None))
    )

st.altair_chart(get_tx_chart(data), True)

## -- end TX ranges

## Merchants

## End Merchants

st.write("Todo")
st.write("1. merchant, category")