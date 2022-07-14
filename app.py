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
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelEncoder

st.title('MSCI 436 Final Project: Detecting Fradulent Transactions')

# @st.cache(allow_output_mutation=True)
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

def preprocess_data(fraud_test):
    # Drop columns with little significance to determining fraud 
    fraud_test.drop(['cc_num', 'first', 'last', 'street', 'trans_num'], axis=1, inplace=True)
    fraud_test.drop(fraud_test.iloc[:,[0]], axis=1, inplace=True)

    # Converting date of birth (dob) to age
    fraud_test['dob'] = pd.to_datetime(fraud_test['dob'])
    fraud_test['age'] = (pd.to_datetime('now') - fraud_test['dob'])/ np.timedelta64(1, 'Y')
    fraud_test['age'] = fraud_test['age'].astype(int)
    fraud_test.drop(['dob'], axis=1, inplace=True)

    # Splitting trans_date_trans_time column into trans_date and trans_time
    fraud_test['trans_date'] = pd.DatetimeIndex(fraud_test['trans_date_trans_time']).date
    fraud_test['trans_time'] = pd.DatetimeIndex(fraud_test['trans_date_trans_time']).time
    fraud_test.drop(['trans_date_trans_time'], axis=1, inplace=True)

    # Transform "merchant" into numeric variable
    label_encoder = LabelEncoder()
    fraud_test.merchant = label_encoder.fit_transform(fraud_test.merchant)

    # Transform "city" into numeric variable
    fraud_test.city = label_encoder.fit_transform(fraud_test.city)

    # Transform "category" into numeric variable
    fraud_test.category = label_encoder.fit_transform(fraud_test.category)

    # Transform "gender" into numeric variable
    fraud_test.gender = fraud_test.gender.map({'M': 1, "F": 0})

    # Transform "state" into numeric variable
    fraud_test.state = label_encoder.fit_transform(fraud_test.state)

    # Transform "job" into numeric variable
    fraud_test.job = label_encoder.fit_transform(fraud_test.job)

    # Convert trans_time into seconds
    fraud_test['trans_date'] =  pd.to_datetime(fraud_test['trans_date'])
    fraud_test.trans_date = fraud_test.trans_date.map(datetime.toordinal)
    fraud_test.trans_time = pd.to_datetime(fraud_test.trans_time,format='%H:%M:%S')
    fraud_test.trans_time = 3600 * pd.DatetimeIndex(fraud_test.trans_time).hour + 60 * pd.DatetimeIndex(fraud_test.trans_time).minute + pd.DatetimeIndex(fraud_test.trans_time).second

    # Seperate target from variables
    X_test = fraud_test.drop('is_fraud', axis=1)
    y_test = fraud_test['is_fraud']
    return X_test, y_test
 

@st.cache(allow_output_mutation=True)
def predict_data(df):   
    # Predict
    model = pickle.load(open('model.pkl', 'rb'))
    y_pred = model.predict(df)
    return y_pred

@st.cache(allow_output_mutation=True)
def format_data(df):
    df['date'] = pd.DatetimeIndex(df['trans_date_trans_time']).date
    df['weekday'] = pd.to_datetime(df['trans_date_trans_time']).dt.day_name()
    df.rename(columns={"long": "lon"}, inplace=True)
    return df[df["is_fraud"]==1]

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is None:
    st.stop()

visualization_data = load_data(uploaded_file)
test_data = visualization_data.copy()

# Get Predictions from Model
test_data, y_test = preprocess_data(test_data)
predictions = predict_data(test_data)

# Get Clean Data For Visualizations
visualization_data["is_fraud"] = predictions ## Add predictions
data = format_data(visualization_data) ## Format and filter transactions that are predicted to be fraudlent

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Time", "Location", "Demographic", "Transaction Amounts", "Merchant"])

### -- Date input
with tab1:
    st.header("Time Visualizations")
    def get_time_series_chart(data):
        hover = alt.selection_single(
            fields=["date"],
            nearest=True,
            on="mouseover",
            empty="none"
        )

        lines = (
            alt.Chart(data, title="Predicted Fradulent Transactions over Time")
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

    st.caption("Filter predicted fradulent transactions by a time period")
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
            alt.Chart(data, title="Weekly view of Predicted Fradulent Transactions")
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
with tab2:
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

with tab3:
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
    st.altair_chart(get_age_chart(data), True)

    ## -- End age ranges

## TX ranges
with tab4:
    st.subheader("Transaction Amount Visualizations")
    def get_tx_chart(data):
        return (
            alt.Chart(data, title="Predicted Fradulent Transactions by Transaction Amount")
            .mark_bar()
            .encode(x=alt.X("amt", bin=True), y="count()", color=alt.Color("amt", legend=None))
        )

    st.altair_chart(get_tx_chart(data), True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Min Transaction Value", round(data["amt"].min(), 2))
    col2.metric("Mean Transaction Value", round(data["amt"].mean(), 2))
    col3.metric("Max Transaction Value", round(data["amt"].max(), 2))

## -- end TX ranges

with tab5:
    st.subheader("Merchant Visualizations")
    ## Merchants

    def get_category_chart(data):
        hover = alt.selection_single(
            fields=["category"],
            nearest=True,
            on="mouseover",
            empty="none"
        )

        lines = (
            alt.Chart(data, title="Predicted Fradulent Transactions by Merchant Category")
            .mark_bar()
            .encode(x="count()", y="category", color=alt.Color("category", legend=None))
        )

        points = lines.transform_filter(hover).mark_circle(size=65)

        tooltips = (
            alt.Chart(data)
            .mark_rule()
            .encode(
                x="count()",
                y="category",
                opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip("category", title="Category"),
                    alt.Tooltip("count()", title="Count"),
                ]
            )
            .add_selection(hover)
        )

        return (lines+points+tooltips).interactive()

    st.altair_chart(get_category_chart(data), True)

    data[["merchant_subscript", "merchant_clean"]] = data["merchant"].str.split("_", n=0, expand=True)
    merchant_df = data.groupby("merchant_clean").agg(count=("merchant_clean", "count")).sort_values("count", ascending=False).reset_index().head(10)
    merchant_df["Percentage of All Transactions (%)"] = merchant_df['count'] / data['merchant'].count() * 100
    merchant_df.rename(columns={"count": "Number of Predicted Fradulent Transactions", "merchant_clean": "Merchant"}, inplace=True)
    st.table(merchant_df)

    ##### Merchant Close Up
    merchant_list = list(merchant_df["Merchant"].head(10).values)
    merchant_list.insert(0, "None")
    merchant_option = st.selectbox("Select a Merchant Name to See More Information", merchant_list)

    if merchant_option != "None":
        st.write(merchant_option)
        merchant_data = data[data["merchant_clean"]==merchant_option]
        merchant_city = merchant_data["city"].values[0]
        merchant_state = merchant_data["state"].values[0]
        merchant_category = merchant_data["category"].values[0]
        merchant_avg_transaction_val = merchant_data["amt"].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("Location", merchant_city + ", " + merchant_state)
        col2.metric("Category", merchant_category)
        col3.metric("Avg Fraud Transaction Amount", round(merchant_avg_transaction_val,2))
    ##### End Merchant Close up

## End Merchants