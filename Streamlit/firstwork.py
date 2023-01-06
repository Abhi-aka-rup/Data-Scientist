import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

# Setting the title
st.title("Total Data Science")

image = Image.open('tdslogo.png')
st.image(image, use_column_width=True)

st.write("### A simple Data App with Streamlit")

#Setting the dataset

dataset_name = st.sidebar.selectbox('Select dataset', ('Breast Cancer', 'Iris', 'Wine'))
classifier_name = st.sidebar.selectbox('Select Classifier', ('SVM', 'KNN'))

def get_dataset(name):
    data = None
   
    match name:
        case 'Iris':
            data = datasets.load_iris()
        case 'Wine':
            data = datasets.load_wine()
        case _:
            data = datasets.load_breast_cancer()
   
    x = data.data
    y = data.target
    
    return x,y

x,y = get_dataset(dataset_name)
st.dataframe(x)
st.write('Shape of your dataset is: ', x.shape)
st.write('Unique target variables: ', len(np.unique(y)))

fig = plt.figure()
sns.boxplot(data=x, orient='h')
st.pyplot(fig)

plt.hist(x)
st.pyplot(fig)

# Building our algorithm

def add_parameter(name_of_clf):
    params = dict()
    match name_of_clf:
        case 'SVM':
            c = st.sidebar.slider('C', 0.01, 15.0)
            params['C'] = c
        case _:
            k = st.sidebar.slider('K', 1, 15)
            params['K'] = k
    return params

params = add_parameter(classifier_name)

# Accessing our classifier

def get_classifier(name_of_clf, params):
    clf = None
    match name_of_clf:
        case 'SVM':
            clf = SVC(C=params['C'])
        case _:
            clf = KNeighborsClassifier(n_neighbors=params['K'])
    return clf

clf = get_classifier(classifier_name, params)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

st.write(y_pred)

accuracy = accuracy_score(y_test, y_pred)

st.write('Classifier name: ', classifier_name)
st.write('Accuracy for your model: ', accuracy)