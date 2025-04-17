# --- Bagian 1: Load semua asset model ---
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder

with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('ordinal_encoder.pkl', 'rb') as f:
    ord_enc = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_list.pkl', 'rb') as f:
    feature_list = pickle.load(f)

# --- Bagian 2: Preprocessing dan Prediksi ---
def preprocess_input(data_df, ord_enc, onehot_encoder, scaler, feature_list):
    data_df['person_gender'] = data_df['person_gender'].str.lower().str.replace(' ', '')
    data_df['person_gender'] = data_df['person_gender'].replace({'male': 'Male', 'female': 'Female'})
    data_df['previous_loan_defaults_on_file'] = data_df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0}).astype(int)
    data_df['person_education'] = ord_enc.transform(data_df[['person_education']])
    
    onehot_cols = ['person_gender', 'person_home_ownership', 'loan_intent']
    encoded = onehot_encoder.transform(data_df[onehot_cols])
    encoded_df = pd.DataFrame(encoded, columns=onehot_encoder.get_feature_names_out(onehot_cols), index=data_df.index)
    
    data_df = data_df.drop(columns=onehot_cols)
    data_df = pd.concat([data_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    num_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    data_df[num_cols] = scaler.transform(data_df[num_cols])

    for col in feature_list:
        if col not in data_df.columns:
            data_df[col] = 0

    return data_df[feature_list]

def predict_loan_status(input_data):
    processed = preprocess_input(input_data, ord_enc, onehot_encoder, scaler, feature_list)
    prediction = model.predict(processed)
    
    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.success("✅ Disetujui")
        return 'Approved'
    else:
        st.error("❌ Ditolak")
        return 'Rejected'


# --- Bagian 3: UI Streamlit ---
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.markdown("Masukkan data peminjam untuk memprediksi apakah pinjaman akan disetujui atau tidak.")

with st.form("loan_form"):
    person_age = st.number_input("Usia", min_value=0.0, max_value=150.0, value=30.0)
    person_gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    person_education = st.selectbox("Pendidikan", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    person_income = st.number_input("Pendapatan Tahunan", min_value=0.0, value=50000.0)
    person_emp_exp = st.number_input("Tahun Pengalaman Kerja", min_value=0, max_value=125, value=5)
    person_home_ownership = st.selectbox("Status Tempat Tinggal", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    loan_amnt = st.number_input("Jumlah Pinjaman", min_value=50.0, max_value=350000.0, value=10000.0)
    loan_intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_int_rate = st.slider("Suku Bunga (%)", min_value=1.0, max_value=25.0, value=12.0)
    cb_person_cred_hist_length = st.slider("Lama Riwayat Kredit (tahun)", 1.0, 40.0, 5.0)
    credit_score = st.slider("Skor Kredit", min_value=300, max_value=850, value=650)
    previous_loan_defaults_on_file = st.selectbox("Pernah Gagal Bayar?", ["Yes", "No"])

    submitted = st.form_submit_button("Prediksi")

    if submitted:
        loan_percent_income = loan_amnt / person_income

        user_input = pd.DataFrame([{
            'person_age': person_age,
            'person_gender': person_gender,
            'person_education': person_education,
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'person_home_ownership': person_home_ownership,
            'loan_amnt': loan_amnt,
            'loan_intent': loan_intent,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }])

        result = predict_loan_status(user_input)