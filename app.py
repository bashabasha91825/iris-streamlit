
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Prepare data and model
data = load_iris()
model1 = RandomForestClassifier()
x = data.data
y = data.target
model1.fit(x, y)

# Streamlit UI
st.header("Iris Flower classification")
sl = st.number_input("Enter sepal Length")
sw = st.number_input("Enter sepal width")
pl = st.number_input("Enter petal Length")
pw = st.number_input("Enter petal width")

# Fixed typo 'prredict' to 'predict' and ensured variable names match
y_pred = model1.predict([[sl, sw, pl, pw]])
op = data.target_names[y_pred[0]]

# Fixed 'st.writer' to 'st.write'
st.write(f"The predicted species is: {op}")
