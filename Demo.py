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
import numpy as np
import datetime

st.set_page_config(page_title='Demo', layout='wide')

encoder_dir = 'model/encoder'
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model('model/xgb_classifier.json')
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('data/new_retail_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

model = load_model()
df = load_data()

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


def calc_freq(data):
    data['Date'] = pd.to_datetime(data['Date'])

    data = data.sort_values(['Customer_ID', 'Date']).copy()
    data['first_purchase'] = data.groupby(['Customer_ID'])['Date'].transform('min') # thay vì dùng aggregate thì nó gom lại kết quả vào group. transform giữ các thành phần trong group và thêm kết quả vào tương tự
    data['amount'] = data.groupby(['Customer_ID']).cumcount() # accumulate count đến row hiện tại trong group. nên cần sort
    data['freq'] = (data['Date'] - data['first_purchase']).dt.days / data['amount'].replace(0, np.nan)
    data['freq'] = data['freq'].astype(float)
    return data

def calc_days_between(data):
    data['prev_purchase'] = data.groupby(['Customer_ID'])['Date'].shift(1)
    data['days_between'] = (data['Date'] - data['prev_purchase']).dt.days
    avg_days = data.groupby(['Customer_ID'])['days_between'].mean().rename('avg_days_between')
    data = data.merge(avg_days, on=['Customer_ID'], how='left')
    return data

def calc_avg_rating(data):
    avg_ratings = data.groupby(['Customer_ID'])['Ratings'].mean().rename('avg_ratings')
    data = data.merge(avg_ratings, on=['Customer_ID'])
    return data

def calc_avg_income(data):
    income_map = {'Low': 1, 'Medium': 2, 'High': 3}
    data['Income_num'] = data['Income'].map(income_map)
    avg_income = data.groupby(['Customer_ID'])['Income_num'].mean().rename('avg_income')
    data = data.merge(avg_income, on=['Customer_ID'])
    def map_income_category(avg):
        if avg < 1.5:
            return 'low'
        elif avg < 2.5:
            return 'medium'
        else:
            return 'high'
    data['avg_income_label'] = data['avg_income'].apply(map_income_category)
    return data

def calc_affinity(data):
    total_purchases = data.groupby('Customer_ID').size()
    category_purchases = data.groupby(['Customer_ID', 'Product_Category']).size().rename('category_count')
    affinity = (category_purchases / total_purchases).rename('category_affinity').reset_index()
    data = data.merge(affinity, on=['Customer_ID', 'Product_Category'], how='left')
    return data

def calc_number_of_days(data):
    if 'first_purchase' in data.columns:
        data = data.drop(columns='first_purchase')
    first_purchase = data.groupby('Customer_ID')['Date'].min().rename('first_purchase')
    data = data.join(first_purchase, on='Customer_ID')
    data['customer_age'] = (data['Date'] - data['first_purchase']).dt.days
    return data

def process_new_row(new_df, data):
    cust_id = new_df['Customer_ID'].values[0]
    customer_data = data[data['Customer_ID'] == cust_id].copy()
    
    
    customer_data = pd.concat([customer_data, new_df], ignore_index=True)
    customer_data['Date'] = pd.to_datetime(customer_data['Date'])
    customer_data = customer_data.sort_values('Date').reset_index(drop=True)

    # Tính freq 
    customer_data = calc_freq(customer_data)

    # Tính days_between và avg_days_between 
    customer_data = calc_days_between(customer_data)

    # Tính avg_ratings 
    customer_data = calc_avg_rating(customer_data)

    # Tính avg_income 
    customer_data = calc_avg_income(customer_data)

    # Tính category_affinity 
    customer_data = calc_affinity(customer_data)
    
    # Tính customer_age
    customer_data = calc_number_of_days(customer_data)

    # Normalize Date (if needed)
    customer_data['Date'] = (customer_data['Date'] - data['Date'].min()).dt.days

    # Trả về chỉ dòng mới đã tính xong
    return customer_data.iloc[[-1]]








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
top_products = df['products'].value_counts().head(15).reset_index()
top_products.columns = ['products', 'count']
_, col2, _ = st.columns([1, 2, 1])
with col2:
    fig = px.bar(top_products, 
                 x='count', 
                 y='products', 
                 orientation='h',
                 color='count', 
    )
    st.plotly_chart(fig)

st.subheader('Bản đồ doanh thu theo quốc gia')
revenue_by_country = df.groupby('Country')['Amount'].sum().reset_index()
fig = px.choropleth(revenue_by_country, locations='Country', locationmode='country names',
                    color='Amount', color_continuous_scale='Viridis')
st.plotly_chart(fig)












st.markdown(
    """
    <h1 style='text-align: center; color: #CCFF00; font-size: 48px; font-family: monospace;'>
        Feature engineering
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h3 style='color: #FFA500; font-family: sans-serif; font-weight: 70;'>
        1. Tần suất mua hàng (freq)
    </h3>
    """,
    unsafe_allow_html=True
)
st.markdown("""
**Công thức:**  
`freq = Số lượng đơn hàng đã mua đến thời điểm hiện tại / (Ngày hiện tại - Ngày mua đầu tiên)`

Tần suất đo mức độ thường xuyên khách hàng quay lại mua sắm. Nếu `freq` tăng (nghĩa là tần suất mua hàng tăng lên), khách hàng tương tác thường xuyên hơn, khả năng **churn thấp**.

**Ví dụ:**  
- Khách hàng A mua đơn hàng đầu tiên cách đây 90 ngày. Hôm nay họ mới mua lại → `freq = 1 / 90 = 0.01` → **khả năng churn cao**.
- Cũng khách hàng A, sau khi mua hôm nay lại tiếp tục mua trong 2 ngày liên tiếp → `freq = 3 / 93 = 0.03` → **tần suất tăng lên → khả năng churn giảm**.
""")

st.markdown(
    """
    <h3 style='color: #FFA500; font-family: sans-serif; font-weight: 70;'>
        2. Khoảng cách trung bình giữa các lần mua (avg_days_between)
    </h3>
    """,
    unsafe_allow_html=True
)
st.markdown("""
**Công thức:**  

`days_between = Số ngày giữa các lần mua hàng` 

`avg_days_between = Trung bình số ngày giữa các lần mua hàng` 

Khoảng cách mua hàng ngắn cho thấy khách hàng có xu hướng mua thường xuyên → **gắn bó**. Khoảng cách dài → dấu hiệu **mất hứng thú**, dễ churn.

**Ví dụ:**  
- Nếu một khách hàng thường cách 5-7 ngày lại mua hàng → khả năng **giữ chân cao**.  
- Nếu khoảng cách trung bình là 60 ngày → **khả năng churn cao**.
""")


st.markdown(
    """
    <h3 style='color: #FFA500; font-family: sans-serif; font-weight: 70;'>
        3. Mức độ gắn bó với danh mục sản phẩm (category_affinity)
    </h3>
    """,
    unsafe_allow_html=True
)
st.markdown("""
**Công thức:**  
`category_affinity = Số lượng đơn thuộc danh mục X / Tổng số đơn hàng của khách`

**Giải thích:**  
Cho biết khách hàng có thiên hướng mua mặt hàng nào. Nếu affinity với danh mục giảm mạnh theo thời gian → **khả năng churn tăng**.

**Ví dụ:**  
- Khách hàng A có 80% đơn thuộc danh mục Clothing → nếu thời gian gần đây không còn mua Clothing nữa → **có thể đã chuyển sang thương hiệu khác**.
""")

# days_since_first
st.markdown(
    """
    <h3 style='color: #FFA500; font-family: sans-serif; font-weight: 70;'>
        4. Tuổi đời khách hàng (days_since_first)
    </h3>
    """,
    unsafe_allow_html=True
)

st.markdown("""
**Công thức:**  
`days_since_first = Ngày hiện tại - Ngày mua hàng đầu tiên của khách hàng`

**Giải thích:**  
Cho biết thời gian khách hàng gắn bó với thương hiệu. Khách hàng mới (`days_since_first` nhỏ) thường dễ churn hơn. Ngược lại, khách hàng lâu năm có xu hướng trung thành hơn. Khách hàng mua 1 năm trước và vẫn quay lại thì sẽ khó rời bỏ.

**Ví dụ:**  
- Khách hàng mới gia nhập 7 ngày → cần chiến dịch giữ chân sớm.  
- Khách hàng cũ 365 ngày → **ít rủi ro churn**.
""")

# Amount
st.markdown(
    """
    <h3 style='color: #FFA500; font-family: sans-serif; font-weight: 70;'>
        5. Features tổng hợp từ số tiền của mỗi đơn hàng
    </h3>
    """,
    unsafe_allow_html=True
)

st.markdown("""
**Công thức:**  
`cum_amount = Tổng số tiền đã chi đến giao dịch hiện tại`  
`cumcount_amt = Số giao dịch tích lũy đến hiện tại (tính từ 1)`  
`avg_amount = cum_amount / cumcount_amt`  
`prev_amount = Số tiền của giao dịch trước`  
`amount_diff = Amount - prev_amount`  

**Giải thích:**  
Các đặc trưng này giúp hiểu rõ hành vi mua hàng theo thời gian:

- `cum_amount`: cho biết khách hàng đã chi bao nhiêu cho từng nhóm sản phẩm
- `cumcount_amt`: đếm số giao dịch đã thực hiện 
- `avg_amount`: tiền trung bình mỗi giao dịch → đo xu hướng chi tiêu tăng hay giảm.  
- `amount_diff`: đo mức chênh lệch giữa giao dịch hiện tại và trước đó → giúp phát hiện tăng/giảm bất thường.  

**Ví dụ:**  
- Một khách hàng mua 5 lần, tổng chi 2 triệu → `avg_amount = 400,000`.  
- Giao dịch mới là 600,000, trước đó là 300,000 → `amount_diff = 300,000` (chi tiêu đang tăng).  
- `cum_amount` tăng đều → khách hàng tiềm năng cao.  
- `amount_diff` âm liên tục → dấu hiệu chi tiêu giảm, có thể sắp churn.
""")












# Dữ liệu
test_accuracy = 0.8082513824045562
confusion_matrix_data = [[20845, 9497], [2085, 27975]]
labels = ["0", "1"]
classification_report = {
    "precision": [0.91, 0.75, "", 0.83, 0.83],
    "recall": [0.69, 0.93, "", 0.81, 0.81],
    "f1-score": [0.78, 0.83, "", 0.81, 0.81],
    "support": [30342, 30060, "", 60402, 60402]
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
    st.image('image/AUC-ROC.png', caption='AUC - ROC curve')
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

with st.expander('Overview of SHAP'):
    st.write('SHAP đo lường mức độ đóng góp của từng feature vào việc xác định giá trị dự đoán cuối cùng.')
    st.markdown('''
    `ϕᵢ = ∑ (|S|! * (M - |S| - 1)! / M!) * [f(S ∪ {i}) - f(S)]`  

    **Trong đó:**  
    i là feature chúng ta đang xét
    - `ϕᵢ`: SHAP value của feature `i`  
    - `S`: tập con các feature không bao gồm `i`  
    - `M`: tổng số feature  
    - `f(S)`: giá trị dự đoán khi chỉ dùng các feature trong `S`  
    - `f(S ∪ {i})`: giá trị dự đoán khi thêm feature mà chúng ta đang quan sát `i` vào tập `S`  
    - `∑`: tính trung bình qua tất cả các tập con `S` có thể
    ''')
    
    st.write('Ý nghĩa là đo mức độ đóng góp của feature i. Ví dụ khi f(S ∪ i) = 0.8 và f(S) = 0.5 thì nghĩa là thêm feature i vào làm tăng dự đoán lên 30%. \
        Và cứ tiếp tục làm thế, duyệt qua toàn bộ tổ hợp các features và xét. Vế nhân đằng trước là cách công thức tính trung bình.')
    st.write('Nhược điểm là phải xét tổ hợp các features nên số lượng tổ hợp sinh ra sẽ lớn và tính toán mất thời gian. Đối với Deep Learning hay Tree thì có các phiên bản gần đúng như TreeSHAP,... dùng riêng để giảm thời gian và chi phí.')
    
with open("model/shap_values.pkl", "rb") as f:
    all_shap_values = joblib.load(f)

shap_values = all_shap_values[:1000]  # chỉ lấy một phần

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

with st.expander('Waterfall plot'):
    st.write('Giúp hiển thị mức độ đóng góp của một feature vào mẫu duy nhất')
    st.write('Bắt đầu từ giá trị base value E[ f(x) ] ~ giá trị trung bình của prediction của tất cả samples.')
    st.write('Các feature sẽ đẩy prediction lên (màu đỏ) hoặc kéo xuống (màu xanh).')
    st.write('Cột y là hiển thị tên feature và giá trị được gán cho feature đó. Giá trị f(x) = 1.46 ở hình chính giữa là kết quả prediction (raw logit chưa qua sigmoid) của model cho feature này')
    

# Force -> động, tương tác được
st.subheader("SHAP Force Plot (1 sample)")
st_shap(shap.plots.force(shap_values[250]), height=300)

with st.expander('Force plot cho 1 sample'):
    st.write('Có thể hiểu đây chính là waterfall plot cho 1 sample được viết gọn lại theo 1 hàng. Giá trị bắt đầu từ điểm ngoài cùng bên trái, feature màu đỏ sẽ đẩy prediction đi lên và màu xanh hạ xuống lại thành kết quả cuối cùng 1.46 với ví dụ waterfall plot ở trên')

# Hiển thị force plot cho nhiều mẫu (interact)
st.subheader("SHAP Force Plot (100 samples)")
st_shap(shap.plots.force(shap_values[0:100]), height=300)
with st.expander('Force plot cho nhiều sample'):
    st.write('Mỗi dòng theo chiều dọc chính là force plot cho 1 sample mà lật lại, quay 90 độ. Giúp hình dung mô hình hoạt động nhất quán không trên nhiều samples.')
    st.write('Phát hiện bias hoặc các feature quá chi phối prediction. Ví dụ như Month đang giảm prediction khá nhiều')
    
# Beeswarm plot -> ảnh tĩnh
st.subheader("SHAP Beeswarm Plot")
_, col2, _ = st.columns([1, 2, 1])
with col2:
    fig2, ax2 = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig2)
with st.expander('Beeswarm plot'):
    st.write('Cho thấy toàn cảnh ảnh hưởng của tất cả feature lên model cho nhiều sample. Mỗi hình tròn (point) chính là một sample')
    st.write('Trục ngang là giá trị SHAP (tức là mức ảnh hưởng của feature tới prediction). Trục dọc là tên các features được xếp theo độ ảnh hưởng trung bình. ')
    st.write('Đỏ: giá trị feature cao. Xanh: giá trị feature thấp. Ví dụ màu đỏ cho freq nghĩa là freq đang nhận giá trị cao và ngược lại')
    st.write(" **Ý nghĩa phân bố điểm:**")
    st.write("   - Nếu nhiều điểm nằm xa trục 0 → feature đó có ảnh hưởng mạnh (âm hoặc dương)")
    st.write("   - Nếu điểm đỏ nằm bên phải → giá trị cao sẽ làm **tăng prediction**.")
    st.write("   - Nếu điểm đỏ nằm bên trái → giá trị cao sẽ làm **giảm prediction**.")

    
st.markdown(
    """
    <h1 style='text-align: center; color: #CCFF00; font-size: 48px; font-family: monospace;'>
        Testing Model
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Một số dữ liệu mẫu")
st.dataframe(df.sample(5))

# Tạo form để người dùng nhập dữ liệu
form = st.form(key='input_form')

# Load giá trị unique
with open("model/unique_categorical_values.json", "r", encoding="utf-8") as f:
    unique_values = json.load(f)

inputs = {}
inputs['Customer_ID'] = form.text_input(f'Enter Customer_ID:', key='Customer_ID')
for col in features:
    if col in ['Age', 'Total_Purchases', 'Amount', 'Ratings']:
        inputs[col] = form.number_input(f'Enter {col}:', value=0, )
    elif col in unique_values:
        options = unique_values[col]
        inputs[col] = form.selectbox(f'Select {col}:', options=options, key=col)
inputs['Date'] = form.date_input('Select Purchase Date', value=datetime.date.today())


submit_button = form.form_submit_button(label='Submit')
if submit_button:
    input_data = pd.DataFrame([inputs])

    input_data = process_new_row(input_data, df)

    st.write("Thông tin khách hàng")
    st.dataframe(input_data)

    encoders = load_encoders()

    transformed_data = transform_data(input_data, encoders)

    st.write("Transformed data")
    
    prediction = model.predict(transformed_data[features])
    st.write(f"Model Prediction: {prediction} == {"churn" if prediction else "not churn"}")






