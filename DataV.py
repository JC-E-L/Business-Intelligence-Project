import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Database connectivity
engine = create_engine('postgresql://postgres:jcladia123456@localhost:5432/online_retail')

# Load data from database tables
sales_data = pd.read_sql_query('SELECT sales_id, invoice_no, quantity, unit_price, productdim_id, customerdim_id, datedim_id FROM sales_fact', con=engine)
product_data = pd.read_sql_query('SELECT productdim_id, stock_code FROM product_dimension', con=engine)
customer_data = pd.read_sql_query('SELECT customerdim_id, country FROM customer_dimension', con=engine)
date_data = pd.read_sql_query('SELECT datedim_id, invoice_date FROM date_dimension', con=engine)

# Calculate total_sales if it's missing
if 'total_sales' not in sales_data.columns:
    sales_data['total_sales'] = sales_data['quantity'] * sales_data['unit_price']

# Merge tables to complete the sales dataset
sales_data = sales_data.merge(product_data, on='productdim_id', how='left')
sales_data = sales_data.merge(customer_data, on='customerdim_id', how='left')
sales_data = sales_data.merge(date_data, on='datedim_id', how='left')

# Convert 'invoice_date' to timezone-naive datetime
sales_data['invoice_date'] = pd.to_datetime(sales_data['invoice_date']).dt.tz_localize(None)

# Sidebar filters
st.sidebar.markdown("<h2 style='color: #636EFA; '>Filter Options</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='color: #FF5733; '>Date</h2>", unsafe_allow_html=True)
date_range = st.sidebar.date_input("Select a Date Range", 
                                   [sales_data['invoice_date'].min(), sales_data['invoice_date'].max()])
st.sidebar.markdown("<h3 style='color: #FF5733;'>Country</h3>", unsafe_allow_html=True)
selected_country = st.sidebar.multiselect('Select a Country', options=sales_data['country'].unique(), 
                                          default=sales_data['country'].unique())
st.sidebar.markdown("<h3 style='color: #FF5733;'>Product</h3>", unsafe_allow_html=True)
selected_product = st.sidebar.multiselect('Select a Product', options=sales_data['stock_code'].unique())

st.sidebar.markdown("<h3 style='color: #4CAF50;'>Price Range</h3>", unsafe_allow_html=True)
price_range = st.sidebar.slider('Select a Price Range', float(sales_data['unit_price'].min()), float(sales_data['unit_price'].max()), 
                                (float(sales_data['unit_price'].min()), float(sales_data['unit_price'].max())))

# Apply filters
filtered_data = sales_data[(sales_data['invoice_date'] >= pd.to_datetime(date_range[0])) & 
                           (sales_data['invoice_date'] <= pd.to_datetime(date_range[1])) & 
                           (sales_data['unit_price'] >= price_range[0]) & 
                           (sales_data['unit_price'] <= price_range[1])]
if selected_country:
    filtered_data = filtered_data[filtered_data['country'].isin(selected_country)]
if selected_product:
    filtered_data = filtered_data[filtered_data['stock_code'].isin(selected_product)]

# Dashboard title
st.markdown("<h1 style='color: #EF553B;'>Online Sales Dashboard</h1>", unsafe_allow_html=True)

# Total Sales and Average Order Values
total_sales = filtered_data['total_sales'].sum()
total_orders = filtered_data['invoice_no'].nunique()
avg_order_value = total_sales / total_orders if total_orders else 0

# Display metrics inline using st.columns
col1, col2 = st.columns(2)
col1.metric("Total Sales", f"${total_sales:,.2f}")
col2.metric("Average Order Value", f"${avg_order_value:,.2f}")

# Group data by product and calculate total sales
top_products = filtered_data.groupby('stock_code')['total_sales'].sum().nlargest(10).reset_index()
top_products.rename(columns={'stock_code': 'Stock Code', 'total_sales': 'Total Sales'}, inplace=True)

# Sort by total sales (ascending or descending based on preference)
top_products = top_products.sort_values(by='Total Sales', ascending=False)

# Create a heatmap-style chart using a color scale
top_products_chart = px.imshow(
    [top_products['Total Sales'].values],  # The data for the heatmap (list of total sales values)
    labels={'x': 'Stock Code', 'y': 'Total Sales'},
    color_continuous_scale='Inferno',  # Choose a color scale for the heatmap
    aspect='auto',  # Automatically adjust the aspect ratio
    title="Top 10 Selling Products Heatmap",
)

# Update layout to fix axis and hover behavior
top_products_chart.update_layout(
    title={'text': 'Top 10 Selling Products Heatmap', 'font': {'color': '#636EFA'}},  # Set the title font color
    xaxis=dict(
        tickmode='array', 
        tickvals=list(range(len(top_products))),  # Set x-axis ticks based on the number of products
        ticktext=top_products['Stock Code'],  # Use the product stock codes as labels on the x-axis
        title='Stock Code',
    ),
    yaxis=dict(
        tickmode='array', 
        tickvals=[0],  # Only one row, so we have a single y-axis label
        ticktext=['Total Sales'],  # This is more relevant as it corresponds to total sales
        title=''
    ),
    coloraxis_colorbar=dict(
        title='Sales Value',  # Colorbar title to indicate that the color represents sales value
    ),
)

# Hover data: Display actual sales values when hovering
top_products_chart.update_traces(
    hovertemplate='<b>%{x}</b><br>Total Sales: %{z}<extra></extra>'  # Show the product code and total sales
)
# Display the heatmap
st.plotly_chart(top_products_chart)

# Group data by country and calculate total sales
sales_by_country = filtered_data.groupby('country')['total_sales'].sum().reset_index()
sales_by_country.rename(columns={'country': 'Country', 'total_sales': 'Total Sales'}, inplace=True)

# Sort data by Total Sales in descending order
sales_by_country = sales_by_country.sort_values(by='Total Sales', ascending=False)

# Create a bar chart with a color scale like a heatmap
sales_by_country_chart = px.bar(
    sales_by_country, 
    x='Country', 
    y='Total Sales', 
    title='Total Sales by Country', 
    color='Total Sales',  # Use Total Sales for color intensity (heatmap-like effect)
    color_continuous_scale='Inferno'  # Color scale similar to heatmap
)

# Update layout for the bar chart
sales_by_country_chart.update_layout(
    title={'text': 'Total Sales by Country', 'font': {'color': '#636EFA'}},  # Set the title font color
    xaxis_title='Country', 
    yaxis_title='Total Sales',
    xaxis_tickangle=-45  # Rotate x-axis labels for better visibility
)

# Display the bar chart
st.plotly_chart(sales_by_country_chart)

# Monthly Sales and Growth Rate Calculation
filtered_data['Month'] = filtered_data['invoice_date'].dt.to_period('M')
monthly_sales = filtered_data.groupby('Month')['total_sales'].sum().to_frame()
monthly_sales.rename(columns={'total_sales': 'Total Sales'}, inplace=True)
monthly_sales.index = monthly_sales.index.to_timestamp()
monthly_sales['Growth Rate'] = monthly_sales['Total Sales'].pct_change() * 100

# Monthly Sales Growth Metric
latest_growth_rate = monthly_sales['Growth Rate'].iloc[-1]
st.metric("Monthly Sales Growth Rate", f"{latest_growth_rate:.2f}%")

# Monthly Sales Trend Chart
monthly_sales_chart = px.line(monthly_sales.reset_index(), x='Month', y='Total Sales', title="Monthly Sales Trend", line_shape="linear", color_discrete_sequence=['#EF553B'])
monthly_sales_chart.update_layout(
    title={'text': 'Monthly Sales Trend', 'font': {'color': '#636EFA'}}
)
st.plotly_chart(monthly_sales_chart)

# Monthly Sales Growth Rate Chart
growth_rate_chart = px.line(monthly_sales.reset_index(), x='Month', y='Growth Rate', title="Monthly Sales Growth Rate", line_shape="linear", color_discrete_sequence=['#EF553B'])
growth_rate_chart.update_layout(
    title={'text': 'Monthly Sales Growth Rate', 'font': {'color': '#636EFA'}}
)
st.plotly_chart(growth_rate_chart)


st.markdown("<h1 style='color: #636EFA;'>Customer Segmentation</h1>", unsafe_allow_html=True)
sales_data = pd.read_sql_query('''
    SELECT 
        s.sales_id, 
        s.invoice_no, 
        s.quantity, 
        s.unit_price, 
        s.total_sales, 
        s.customerDim_id, 
        s.dateDim_id, 
        d.invoice_date, 
        c.customer_id, 
        c.country
    FROM sales_fact s
    JOIN date_dimension d ON s.dateDim_id = d.dateDim_id
    JOIN customer_dimension c ON s.customerDim_id = c.customerDim_id
    ''', con=engine)

# Calculate Recency, Frequency, and Monetary value
reference_date = sales_data['invoice_date'].max()  # Last date in dataset

rfm = sales_data.groupby('customer_id').agg({
    'invoice_date': lambda x: (reference_date - x.max()).days,  # Recency in days
    'invoice_no': 'nunique',  # Frequency (number of purchases)
    'total_sales': 'sum'  # Monetary (total spend)
}).reset_index()

rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

# Standardize the features for clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

# Apply K-Means clustering (let's assume 4 clusters for now)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

# Create a scatter plot with Frequency (y-axis) and Monetary (x-axis)
fig = px.scatter(
    rfm, 
    x='monetary',  
    y='frequency', 
    color='cluster',
    title="Frequency vs. Monetary Value",
    labels={'frequency': 'Number of Transactions (Frequency)', 'monetary': 'Total Spent (Monetary Value)'},
    color_continuous_scale=px.colors.sequential.Inferno
)

fig.update_layout(title={'font': {'color': '#636EFA'}})
st.plotly_chart(fig)

# Predictive Analysis with Linear Regression
st.markdown("<h1 style='color: #636EFA;'>Sales Forecasting</h1>", unsafe_allow_html=True)
filtered_data['Month'] = filtered_data['invoice_date'].dt.to_period('M')
monthly_sales = filtered_data.groupby('Month')['total_sales'].sum().to_frame()
monthly_sales.index = monthly_sales.index.to_timestamp()

# Prepare data for linear regression
monthly_sales['MonthIndex'] = np.arange(len(monthly_sales))
X = monthly_sales[['MonthIndex']]
y = monthly_sales['total_sales']

# Fit the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
monthly_sales['predicted_sales'] = lin_reg.predict(X)

# Plot the actual and predicted sales
fig_forecast = px.line(monthly_sales, x=monthly_sales.index, y=['total_sales', 'predicted_sales'],
                       labels={'value': 'Sales', 'variable': 'Type'})
fig_forecast.update_layout(
    title={'text':'Sales Forecasting with Linear Regression', 'font': {'color': '#EF553B'}}
)
st.plotly_chart(fig_forecast)

# Forecast future sales
future_months = pd.DataFrame({'MonthIndex': np.arange(len(monthly_sales), len(monthly_sales) + 12)})
future_sales = lin_reg.predict(future_months)
future_sales_df = pd.DataFrame({'Month': pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=12, freq='ME'), 'Predicted Sales': future_sales})

fig_future_sales = px.line(future_sales_df, x='Month', y='Predicted Sales')
fig_future_sales.update_layout(
    title={'text':'Future Sales Prediction (Next 12 Months)', 'font': {'color': '#EF553B'}}
)
st.plotly_chart(fig_future_sales)
