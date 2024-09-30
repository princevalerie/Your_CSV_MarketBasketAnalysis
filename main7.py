import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Function to detect delimiter
def detect_delimiter(file):
    sample = file.getvalue().decode('utf-8').splitlines()[:5]
    comma_count = sum([line.count(',') for line in sample])
    semicolon_count = sum([line.count(';') for line in sample])
    return ',' if comma_count >= semicolon_count else ';'

# Application title
st.title('Market Basket Analysis')

# Upload file CSV or Excel
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

# Initialize session state variables
if 'product_association' not in st.session_state:
    st.session_state.product_association = None
if 'unique_items' not in st.session_state:
    st.session_state.unique_items = None
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None

if uploaded_file is not None:
    try:
        # Detect delimiter
        if uploaded_file.name.endswith('.csv'):
            delimiter = detect_delimiter(uploaded_file)
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, sep=delimiter, on_bad_lines='skip')
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)

        st.write("Data yang diunggah:")
        st.dataframe(data.head())
        
        # Selecting 'Order ID' column
        order_id_column = st.selectbox("Pilih kolom untuk Order Id / Id Transaksi:", data.columns)
        
        # Selecting 'Item Name' column
        item_column = st.selectbox("Pilih kolom untuk Nama Barang:", data.columns)
        
        # Preprocessing: Remove duplicates and NaN values in selected columns
        data.drop_duplicates(inplace=True)
        data = data.dropna(subset=[order_id_column, item_column])
        
        # Set confidence threshold with step 0.1
        confidence_threshold = st.slider("Atur threshold untuk Confidence:", 0.5, 1.0, 0.6, step=0.1)

        # Transform data to appropriate format for analysis
        basket = data.groupby([order_id_column, item_column])[item_column].count().unstack().reset_index().fillna(0).set_index(order_id_column)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        # Calculate frequent itemsets using apriori
        frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
        
        # Calculate association rules based on confidence
        product_association = association_rules(frequent_itemsets, metric='confidence', min_threshold=confidence_threshold)
        product_association = product_association.sort_values(['support', 'confidence'], ascending=[False, False]).reset_index(drop=True)

        # Save results to session state
        st.session_state.product_association = product_association
        # Extract unique items for selection
        st.session_state.unique_items = [str(item) for item in product_association['antecedents'].unique()]

        # # Setelah menghitung aturan asosiasi
        # Convert frozensets to lists for easier manipulation
        product_association['antecedents'] = product_association['antecedents'].apply(lambda x: list(x)[0])
        product_association['consequents'] = product_association['consequents'].apply(lambda x: list(x)[0])

        # Select item for consequent items
        selected_item = st.selectbox("Pilih item untuk melihat barang consequent:", product_association['antecedents'].unique())

        # Filter rules based on the selected item
        selected_item_rules = product_association[
            product_association['antecedents'].apply(lambda x: selected_item in x)
        ]

        # Display the top recommendation if available
        if not selected_item_rules.empty:
            # Only keep the top 1 recommendation
            top_recommendation = selected_item_rules.iloc[0]
            st.success(f'Hasil Rekomendasi untuk {selected_item}:')
            st.write("Orang yang membeli :", top_recommendation['antecedents'])
            st.write("Biasanya juga membeli :", top_recommendation['consequents'])
        else:
            st.subheader(f'Tidak ada aturan ditemukan untuk {selected_item}.')

        # Optionally, show the complete analysis results if needed
        st.write("Hasil Analisis Lengkap:")
        st.dataframe(product_association[['antecedents', 'consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'leverage', 'conviction', 'zhangs_metric']])

    except Exception as e:
        st.error(f"Tidak Ada Hasil Analisis: {e}")
