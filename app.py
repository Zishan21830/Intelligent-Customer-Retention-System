import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st

# load model
model = tf.keras.models.load_model('model.keras')

# load the encoders and Scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encode_gender = pickle.load(file)
    
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scalers = pickle.load(file)

st.title('Customer Churn Prediction')


geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encode_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 1, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# input data
input_data = {
    'CreditScore' : [credit_score],
    'Gender' : [label_encode_gender.transform([gender])[0]],
    'Age'  : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
}
input_df = pd.DataFrame(input_data)

# Encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine features
input_combined = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input
input_scaled = scalers.transform(input_combined)


# predict churn
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

st.write(f' Churn Probability {prediction_prob}')
if prediction_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')