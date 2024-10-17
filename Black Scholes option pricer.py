#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns 
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy import log,exp,sqrt
import plotly.graph_objects as go 


# In[22]:


st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)


class blackscholes:
    def __init__(
        self,
        time_to_maturity:float,
        strike:float,
        volatility:float,
        current_price:float,
        interest_rate:float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.volatility = volatility
        self.current_price = current_price
        self.interest_rate = interest_rate
    
    def calc_prices(
        self,
    ):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        volatility = self.volatility
        current_price = self.current_price
        interest_rate = self.interest_rate
    
        d1 = (
        log(current_price/strike) + 
        (interest_rate + 0.5*volatility**2) * time_to_maturity)/(
        volatility * sqrt(time_to_maturity))
    
        d2 = d1 - volatility * sqrt(time_to_maturity)
    
        call_price = current_price * norm.cdf(d1) - (norm.cdf(d2) * strike * exp(-(interest_rate * time_to_maturity)))
    
        put_price = (norm.cdf(-d2) * strike * exp(-(interest_rate * time_to_maturity))) - current_price * norm.cdf(-d1)
    
        self.call_price = call_price
        self.put_price = put_price
    
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)
    
        self.call_gamma = norm.cdf(d1)/ (strike * volatility * sqrt(time_to_maturity))
        self.put_gamma = self.call_gamma
    
        return call_price, put_price

with st.sidebar:
    st.title('Black Scholes Options Pricer')
    
    current_price = st.number_input("Current Asset Price", value = 100.00)
    time_to_maturity = st.number_input('Time to Maturity (years)', value = 1.00)
    strike = st.number_input('Strike Price', value = 100.00)
    volatility = st.number_input('Volatility(sigma)', value = 0.25 )
    interest_rate = st.number_input('Interest Rate', value = 0.10)
    
    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    spot_min = st.number_input('Min Spot price',value = current_price*0.8, min_value = 0.01, step = 0.01)
    spot_max = st.number_input('Max Spot Price',value = current_price*1.2, min_value = 0.01, step = 0.01)
    vol_min = st.slider('Min Volatility for Heatmap',min_value = 0.01, max_value = 1.00, value = volatility*0.5, step = 0.01)
    vol_max = st.slider('Max Volatility for Heatmap',min_value = 0.01, max_value = 1.00, value = volatility*1.5, step = 0.01)
    
    spot_range = np.linspace(spot_min,spot_max,10)
    vol_range = np.linspace(vol_min,vol_max,10)
    

def heatmap_plot(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(spot_range),len(vol_range)))
    put_prices = np.zeros((len(spot_range),len(vol_range)))
    
    for i,vol in enumerate(vol_range):
        for j,vol in enumerate(spot_range):
            bs_temp = blackscholes(
            time_to_maturity = bs_model.time_to_maturity,
            strike = bs_model.strike,
            current_price = bs_model.current_price,
            volatility = bs_model.volatility,
            interest_rate = bs_model.interest_rate
            )
            bs_temp.calc_prices()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price

    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put

st.title('Black Scholes Options Pricing Model')

input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

bs_model = blackscholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calc_prices()

col1, col2 = st.columns([1,1], gap="small")


with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = heatmap_plot(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    _, heatmap_fig_put = heatmap_plot(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_put)

