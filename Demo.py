import streamlit as st
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_shap import st_shap
import shap
import json
import plotly.express as px
import calendar

st.set_page_config(page_title='Demo', layout='wide')

model = XGBClassifier()
model.load_model('model/xgb_classifier.json')
encoder_dir = 'model/encoder'
df = pd.read_csv('data/new_retail_data.csv')


features = [
    'City', 'Country', 'Age', 'Gender', 'Income', 'Customer_Segment', 'Total_Purchases',
    'Amount', 'Product_Category', 'Product_Brand', 'Product_Type', 'Shipping_Method',
    'Payment_Method', 'Ratings', 'Month', 'Year', 'freq', 'avg_days_between', 'days_between',
    'avg_ratings', 'avg_income', 'avg_income_label', 'category_affinity', 'customer_age'
]

categorical_cols = [
    'City', 'Country', 'Gender', 'Income', 'Customer_Segment', 'Product_Category', 
    'Product_Brand', 'Product_Type', 'Shipping_Method', 'Payment_Method', 'Month', 'avg_income_label'
]

def load_encoders():
    encoders = {}
    for col in categorical_cols:
        try:
            le = joblib.load(os.path.join(encoder_dir, f'{col}_label_encoder.pkl'))
            encoders[col] = le
        except FileNotFoundError:
            st.warning(f"Encoder for {col} not found!")
    return encoders

def transform_data(input_data, encoders):
    for col in categorical_cols:
        if col in input_data.columns:
            if col in encoders:
                le = encoders[col]
                input_data[col] = le.transform(input_data[col].fillna('unknown')) 
            else:
                st.warning(f"Encoder for {col} not available.")
    return input_data




st.markdown(
    """
    <h1 style='text-align: center; color: #CCFF00; font-size: 48px; font-family: monospace;'>
        Exploratory data analysis 
    </h1>
    """,
    unsafe_allow_html=True
)
st.subheader('Số lượng transaction của mỗi category ở từng counrtry')
top_products_by_country = df.groupby(['Country', 'Product_Category']).size().reset_index(name='Count')
st.bar_chart(top_products_by_country.pivot(index='Product_Category', columns='Country', values='Count').fillna(0))
st.write('Chúng ta có thể thấy là mặt hàng Grocery được tiêu thụ ở Mỹ nhiều nhất và ít nhất Canada.')
st.write('Và Mỹ có nhiều đơn hàng về Grocery và ít đơn về Home Decor')
st.write('Hoặc có thể xem dưới dạng tỉ trọng phần trăm như bên dưới')

categories = df['Product_Category'].dropna().unique()

# Chia mỗi 6 pie chart thành 1 hàng
for i in range(0, len(categories), 6):
    row_cats = categories[i:i+6]
    cols = st.columns(len(row_cats))  # chỉ tạo số cột đúng bằng số pie còn lại
    for j, cat in enumerate(row_cats):
        cat_df = df[df['Product_Category'] == cat]
        country_counts = cat_df['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']

        fig = px.pie(
            country_counts,
            names='Country',
            values='Count',
            title=f'{cat}'
        )
        cols[j].plotly_chart(fig, use_container_width=True)

st.subheader('Sản phẩm theo độ tuổi: độ tuổi, số lượng tiêu thụ, danh mục sản phẩm')
bins = [0, 18, 25, 35, 45, 55, 65, 100]
labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']
df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
age_product = df.groupby(['age_group', 'Product_Category']).size().reset_index(name='Count')
fig = px.scatter(
    age_product, 
    x='age_group', 
    y='Count', 
    color='Product_Category', 
    size='Count',  # Kích thước điểm dựa trên số lượng
    size_max=25,   # Kích thước tối đa của điểm
)
st.plotly_chart(fig)

st.subheader('Sản phẩm bán chạy theo mùa')
season_df = df.groupby(['Year', 'Month', 'Product_Category']).size().reset_index(name='Count')
years = sorted(season_df['Year'].unique())
color_palettes = [px.colors.qualitative.Prism, px.colors.qualitative.Pastel]
month_order = list(calendar.month_name)[1:]  # ['January', 'February', …, 'December']
season_df['Month'] = pd.Categorical(season_df['Month'],
                                    categories=month_order,
                                    ordered=True)

for i, year in enumerate(sorted(years)):
    year = int(year)
    st.markdown(f"### Năm {year}")
    year_df = season_df[season_df['Year'] == year]
    fig = px.bar(
        year_df,
        x='Month',
        y='Count',
        color='Product_Category',
        barmode='group',
        color_discrete_sequence=color_palettes[i % len(color_palettes)]
    )
    st.plotly_chart(fig)

st.subheader('Top 15 sản phẩm bán chạy')
top_products = df['products'].value_counts().head(10).reset_index()
top_products.columns = ['products', 'count']
_, col2, _ = st.columns([1, 2, 1])
with col2:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='count', y='products', data=top_products, palette='rocket', ax=ax)
    ax.set_title('Top 10 Frequently Purchased Products')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Product')
    st.pyplot(fig)

st.subheader('Bản đồ doanh thu theo quốc gia')
revenue_by_country = df.groupby('Country')['Amount'].sum().reset_index()
fig = px.choropleth(revenue_by_country, locations='Country', locationmode='country names',
                    color='Amount', title='Doanh thu theo quốc gia', color_continuous_scale='Viridis')
st.plotly_chart(fig)







# Dữ liệu
test_accuracy = 0.8434397838948642
confusion_matrix_data = [[27916, 5083], [4364, 22978]]
labels = ["0", "1"]
classification_report = {
    "precision": [0.86, 0.82, "", 0.84, 0.84],
    "recall": [0.85, 0.84, "", 0.84, 0.84],
    "f1-score": [0.86, 0.83, "", 0.84, 0.84],
    "support": [32999, 27342, "", 60341, 60341]
}
report_index = ["0", "1", "", "macro avg", "weighted avg"]


st.markdown(
    """
    <h1 style='text-align: center; color: #CCFF00; font-size: 48px; font-family: monospace;'>
        Model evaluation report
    </h1>
    """,
    unsafe_allow_html=True
)
# Accuracy
st.metric(label="Test Accuracy", value=f"{test_accuracy:.4f}")

# Confusion matrix
st.subheader("Confusion Matrix")
_, col2, _ = st.columns([1, 2, 1])
with col2:
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

# Classification 
st.subheader("Classification Report")

df_report = pd.DataFrame(classification_report, index=report_index)
st.dataframe(df_report.style.format(precision=2), use_container_width=True)

_, col2, _ = st.columns(3)
with col2:
    st.image('image/AUC-ROC.jpg', caption='AUC - ROC curve')
_, col2, _ = st.columns([1, 2, 1])
with col2:
    st.image('image/confusion.jpg', caption='Normalized confusion matrix')
_, col2, _ = st.columns(3)
with col2:
    st.image('image/importance.jpg', caption='Feature importance by built-in XGBoost function')
with st.expander('Hạn chế của cách đo feature importance này'):
    st.write('Thực ra có 3 loại feature importance: cover, gain và weight. Mỗi loại cho 1 chart khác nhau nên khó đánh giá tổng quan và cần hiểu rõ cách tính toán để biết được ý nghĩa thật sự của từng loại.')
    st.write('Các biểu đồ feature importance thường chỉ cho thấy **feature nào quan trọng hơn**, nhưng không cho biết '
             '**nó ảnh hưởng đến phân loại như thế nào** (về phía nhãn nào, theo hướng tăng hay giảm).')
    st.write('Ví dụ nếu mô hình có 3 lớp: `Dog`, `Cat`, `Bird`, thì biểu đồ feature importance không nói rõ feature X '
                'giúp phân biệt `Cat` với `Dog` hay `Cat` với `Bird`.')
    st.write('Ngoài ra, biểu đồ feature importance cũng không cho biết **mức độ ảnh hưởng cụ thể lên dự đoán**. '
                'Ví dụ nếu mô hình dự đoán xác suất ban đầu là 0.5 thì không biết feature X có đóng góp thêm 0.3 '
                '(~30%) cho nhãn 1 hay không')
    st.write("Vì vậy cần những cách đánh giá khác như SHAP.")





# Feature Importance
st.markdown(
    """
    <h1 style='text-align: center; color: #CCFF00; font-size: 48px; font-family: monospace;'>
        SHAP - SHapley Additive exPlantions
    </h1>
    """,
    unsafe_allow_html=True
)
shap_values = joblib.load('model/shap_values.pkl')

# Waterfall plot -> ảnh tĩnh
st.subheader("SHAP Waterfall Plot")
col1, col2, col3 = st.columns(3)
with col1:
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[250], show=False)
    st.pyplot(fig)
with col3:
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[500], show=False)
    st.pyplot(fig)
    


# Force -> động, tương tác được
st.subheader("SHAP Force Plot (1 sample)")
st_shap(shap.plots.force(shap_values[0]), height=300)

# Hiển thị force plot cho nhiều mẫu (interact)
st.subheader("SHAP Force Plot (100 samples)")
st_shap(shap.plots.force(shap_values[0:100]), height=300)


# Beeswarm plot -> ảnh tĩnh
st.subheader("SHAP Beeswarm Plot")
_, col2, _ = st.columns([1, 2, 1])
with col2:
    fig2, ax2 = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig2)


st.markdown(
    """
    <h1 style='text-align: center; color: #CCFF00; font-size: 48px; font-family: monospace;'>
        Testing Model
    </h1>
    """,
    unsafe_allow_html=True
)

# Tạo form để người dùng nhập dữ liệu
form = st.form(key='input_form')

# Load giá trị unique
with open("model/unique_categorical_values.json", "r", encoding="utf-8") as f:
    unique_values = json.load(f)
    
inputs = {}
for col in features:
    if col in ['Age', 'Total_Purchases', 'Amount', 'Ratings']:
        inputs[col] = form.number_input(f'Enter {col}:', value=0)
    elif col in unique_values:
        options = unique_values[col]
        inputs[col] = form.selectbox(f'Select {col}:', options=options, key=col)
    else:
        inputs[col] = form.text_input(f'Enter {col}:', key=col)

submit_button = form.form_submit_button(label='Submit')
if submit_button:
    input_data = pd.DataFrame([inputs])

    encoders = load_encoders()

    transformed_data = transform_data(input_data, encoders)

    st.write("Transformed data")
    
    prediction = model.predict(transformed_data)
    st.write("Model Prediction:")
    st.write(prediction)