import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

st.title('Aplikasi Prediksi Regression Dataset')

# Load Dataset
with st.expander('Dataset'):
    data = pd.read_csv('Regression.csv')
    st.write(data)

    st.success('Informasi Dataset')
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.success('Analisis Univariat')
    deskriptif = data.describe()
    st.write(deskriptif)

# Data Preprocessing
with st.expander('Preprocessing Data'):
    st.success('Hapus Outlier')
    
    def remove_outlier(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data = remove_outlier(data, col)
    
    st.write(f'Dataset setelah outlier removal: {data.shape}')
    st.write(data.head())


# Visualisasi Data
with st.expander('Korelasi Heatmap Setelah Preprocessing'):
    st.info('Korelasi Antar Fitur Numerik')
    
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = data[numeric_cols].corr()
    
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', fmt='.2f')
    plt.title('Korelasi Antar Fitur')
    st.pyplot(fig)

with st.expander('Visualisasi Per Kolom Setelah Preprocessing'):
    st.info('Distribusi Data untuk Kolom Numerik')
    
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        st.write(f'Visualisasi untuk kolom: {col.capitalize()}')
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, color='blue')
        plt.xlabel(col.capitalize())
        plt.title(f'Distribusi Kolom {col.capitalize()}')
        st.pyplot(fig)
    
    st.info('Distribusi Data untuk Kolom Kategori yang Sudah Di-encode')
    categorical_cols = [col for col in data.columns if col not in numeric_cols]
    for col in categorical_cols:
        st.write(f'Visualisasi untuk kolom: {col.capitalize()}')
        fig, ax = plt.subplots()
        sns.barplot(x=data[col].value_counts().index, y=data[col].value_counts().values, palette='Set2')
        plt.xlabel(col.capitalize())
        plt.ylabel('Count')
        plt.title(f'Distribusi Kolom {col.capitalize()}')
        st.pyplot(fig)

# Modelling
with st.expander('Modelling'):
    st.info('Preprocessing Data untuk Model')
    
    # Pilih kolom numerik dan non-numerik
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Encoding untuk kolom kategori
    if not categorical_cols.empty:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    st.write('Dataset setelah encoding:', data.head())
    
    # Pisahkan fitur dan target
    target = st.selectbox('Pilih Target Kolom', options=data.columns)
    features = data.drop(columns=[target])
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, data[target], test_size=0.3, random_state=42
    )
    st.write(f'Training set: {X_train.shape}')
    st.write(f'Testing set: {X_test.shape}')
    
    # Melatih Model
    st.success('Train Random Forest Regressor')
    rf_model = RandomForestRegressor(max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f'Mean Absolute Error: {mae}')

# Prediction
with st.sidebar:
    st.header('Prediksi Data Baru')
    input_features = {}
    for col in features.columns:
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        mean_val = float(data[col].mean())
        input_features[col] = st.slider(
            f'{col}', min_value=min_val, max_value=max_val, value=mean_val
        )

    st.write('Input Data:', input_features)

with st.expander('Hasil Prediksi'):
    new_data = pd.DataFrame([input_features])
    prediction = rf_model.predict(new_data)
    st.write('Hasil Prediksi:', prediction[0])
