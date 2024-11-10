import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

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
date_range = st.sidebar.date_input("Select Date Range", 
                                   [sales_data['invoice_date'].min(), sales_data['invoice_date'].max()])
st.sidebar.markdown("<h3 style='color: #FF5733;'>Select Country</h3>", unsafe_allow_html=True)
selected_country = st.sidebar.multiselect('Select Country',options=sales_data['country'].unique(), 
                                          default=sales_data['country'].unique())
st.sidebar.markdown("<h3 style='color: #FF5733;'>Select Product</h3>", unsafe_allow_html=True)
selected_product = st.sidebar.multiselect('Select Product',options=sales_data['stock_code'].unique())

st.sidebar.markdown("<h3 style='color: #4CAF50;'>Select Price Range</h3>", unsafe_allow_html=True)
price_range = st.sidebar.slider('Select Price Range',float(sales_data['unit_price'].min()), float(sales_data['unit_price'].max()), 
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
st.markdown("<h1 style='color: #636EFA;'>Interactive Sales Dashboard</h1>", unsafe_allow_html=True)


#Total Sales and Average Order Values
total_sales = filtered_data['total_sales'].sum()
total_orders = filtered_data['invoice_no'].nunique()
avg_order_value = total_sales / total_orders if total_orders else 0

# Display metrics inline using st.columns
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Sales", f"${total_sales:,.2f}")

with col2:
    st.metric("Average Order Value", f"${avg_order_value:,.2f}")

# Top 10 Selling Products Bar Chart
top_products = filtered_data.groupby('stock_code')['total_sales'].sum().nlargest(10).reset_index()
top_products.rename(columns={'stock_code': 'Stock Code', 'total_sales': 'Total Sales'}, inplace=True)
top_products_chart = px.bar(top_products, x='Stock Code', y='Total Sales', title='Top 10 Selling Products', color_discrete_sequence=['#EF553B'])
top_products_chart.update_layout(
    title={'text': 'Top 10 Selling Products', 'font': {'color': '#636EFA'}}  # Set the title font color
)
st.plotly_chart(top_products_chart)

# Sales by Country Bar Chart
sales_by_country = filtered_data.groupby('country')['total_sales'].sum().reset_index()
sales_by_country.rename(columns={'country': 'Country', 'total_sales': 'Total Sales'}, inplace=True)
sales_by_country_chart = px.bar(sales_by_country, x='Country', y='Total Sales', title='Total Sales by Country',  color_discrete_sequence=['#EF553B'])
sales_by_country_chart.update_layout(
    title={'text': 'Total Sales by Country', 'font': {'color': '#636EFA'}}  # Set the title font color
)
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

