# Lab12: Classification des fleurs Iris
# Realisé par Aymane Heddad Emsi 2023 2024
# Import des pakages
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import streamlit as st
# Step1: Dataset
iris = datasets.load_iris()
print(iris.data)
print(iris.target)
print(iris.target_names)
# Step2: Model
model = RandomForestClassifier()
# Step3: Train
model.fit(iris.data,iris.target)
# Step4: Test


# Model Deployment on "streamlit run Lab12_aymane_heddad.py"
st.header('Classification des fleurs Iris')
def user_input():
    sepal_length = st.sidebar.slider("sepal length",4.3,7.9,6.0)
    sepal_width = st.sidebar.slider("sepal width", 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider("petal length", 1.0, 9.2, 2.0)
    petal_width = st.sidebar.slider("petal width", 0.1, 2.5, 1.0)
    data = {
        'sepal_length':sepal_length,
        'sepal_width':sepal_width,
        'petal_length':petal_length,
        'petal_width':petal_width
    }
    flower_features = pd.DataFrame(data, index= [0])
    return flower_features

df = user_input()
st.write(df)
st.subheader("Iris flower Prediction")
prediction = model.predict(df)
st.write(iris.target_names[prediction])
p= iris.target_names[prediction]
print(p)
st.image(f'{iris.target_names[prediction][0]}.jpg',width=300)



