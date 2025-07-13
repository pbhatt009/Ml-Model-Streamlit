import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def show_iris():
    # Upload iris.csv file
    df = pd.read_csv("pages/iris/iris.csv")
    # st.dataframe(df.head())
    # st.write(df.describe())

    # create the model
    xtrain,xtest,ytrain,ytest=train_test_split(df.drop('species',axis=1),df['species'],test_size=0.2,random_state=1)
    clf=LogisticRegression()
    clf.fit(xtrain,ytrain)
    pred=clf.predict(xtest)


    ## ui

    st.title("🌸 Iris Classifier")
    sepal_length = st.slider("Sepal Length", min_value=4.0, max_value=8.0,step=0.1)
    sepal_width = st.slider("Sepal Width", min_value=1.0, max_value=5.0,step=0.1)
    petal_length = st.slider("Petal Length", min_value=1.0, max_value=8.0,step=0.1)
    petal_width = st.slider("Petal Width", min_value=0.0, max_value=3.0,step=0.1)
    input=[sepal_length,sepal_width,petal_length,petal_width]
    flower=clf.predict(np.array(input).reshape(1,4))[0].upper()
    st.success(f'The Flower is: {flower}')
    if flower == "SETOSA":
        st.image("pages/iris/setosa.jpeg", width=500, caption="Iris Setosa")

    elif flower == "VERSICOLOR":
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", width=500, caption="Iris Versicolor")
    elif flower == "VIRGINICA":
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", width=500, caption="Iris Virginica")
