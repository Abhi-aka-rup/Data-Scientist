import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection

matplotlib.use('Agg')

from PIL import Image

st.title("Total Data Science")
image = Image.open('tdslogo.png')
st.image(image, use_column_width=True)
fig = plt.figure()

def main():
    activities = ['EDA', 'Visualisation', 'model', 'About us']
    option = st.sidebar.selectbox('Selection option:', activities)

    match option:
        case 'EDA':
            st.subheader('Exploratory Data Analysis')
            data = st.file_uploader('Upload dataset:', type=['csv', 'xlsx', 'txt', 'json'])
            st.success('Data successfully loaded')
            if data is not None:
                df = pd.read_csv(data)
                st.dataframe(df.head(50))
                
                if st.checkbox("Display shape"):
                    st.write(df.shape)
                if st.checkbox("Display columns"):
                    st.write(df.columns)
                if st.checkbox("Select multiple columns"):
                    selected_columns = st.multiselect('Select preferred columns: ', df.columns)
                    df1 = df[selected_columns]
                    st.dataframe(df1)
                    
                if st.checkbox("Display summary"):
                    st.write(df1.describe().T)
                
                if st.checkbox('Display null values'):
                    st.write(df.isnull().sum())
                    
                if st.checkbox('Display the data types'):
                    st.write(df.dtypes)
                    
                if st.checkbox('Display Correlation of data various columns'):
                    st.write(df.corr())
        
        case 'Visualisation':
            st.subheader('Visualisation')
            data = st.file_uploader('Upload dataset:', type=['csv', 'xlsx', 'txt', 'json'])
            st.success('Data successfully loaded')
            if data is not None:
                df = pd.read_csv(data)
                st.dataframe(df.head(50))
                
            if st.checkbox('Select multiple columns to plot'):
                selected_columns = st.multiselect('Select preferred columns: ', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)
                
            if st.checkbox('Display Heatmap'):
                fig = plt.figure()
                sns.heatmap(df1.corr(), vmax=1, square=True, annot=True, cmap='viridis')
                st.pyplot(fig)
                
            if st.checkbox('Display Pairplot'):
                fig = sns.pairplot(df1, diag_kind='kde')
                st.pyplot(fig)
                
            if st.checkbox('Display Pie Chart'):
                all_columns = df.columns.to_list()
                pie_columns = st.selectbox('Select column to display', all_columns)
                fig = plt.figure()
                pieChart = df[pie_columns].value_counts().plot.pie(autopct='%1.1f%%')
                st.pyplot(fig)
        
        case 'model':
            st.subheader('Model Building')
            data = st.file_uploader('Upload dataset:', type=['csv', 'xlsx', 'txt', 'json'])
            st.success('Data successfully loaded')
            if data is not None:
                df = pd.read_csv(data)
                st.dataframe(df.head(50))
                
                if st.checkbox('Select Multiple Columns'):
                    new_data = st.multiselect('Select your preferred columns', df.columns)
                    df1 = df[new_data]
                    st.dataframe(df1)
                    
                    X = df1.iloc[:, 0:-1]
                    y = df1.iloc[:, -1]
                    
                seed = st.sidebar.slider('Seed', 1, 200)
                classifier_name = st.sidebar.selectbox('Select you preferred classifier:', ('KNN', 'SVM', 'LR', 'Naive Bayes', 'Decision Tree'))
                
                def add_parameter(name_of_clf):
                    params = dict()
                    match name_of_clf:
                        case 'KNN':
                            K = st.sidebar.slider('K', 1, 15)
                            params['K'] = K
                        case 'SVM':
                            C = st.sidebar.slider('C', 0.01, 15.0)
                            params['C'] = C
                    return params
                
                params = add_parameter(classifier_name)
                
                def get_classifier(name_of_clf, params):
                    clf = None
                    match name_of_clf:
                        case 'SVM':
                            clf = SVC(C=params['C'])
                        case 'KNN':
                            clf = KNeighborsClassifier(n_neighbors=params['K'])
                        case 'LR':
                            clf = LogisticRegression()
                        case 'Naive Bayes':
                            clf = GaussianNB()
                        case 'Decision Tree':
                            clf = DecisionTreeClassifier()
                        case _:
                            st.warning('Select your choice of algorithm')
                    return clf
                
                clf = get_classifier(classifier_name, params)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
                
                clf.fit(X_train, y_train)
                
                y_pred = clf.predict(X_test)
                st.write('Predictions: ', y_pred)
                
                accuracy = accuracy_score(y_test, y_pred)
                st.write('Classifier name: ', classifier_name)
                st.write('Accuracy: ', accuracy)
                
        case 'About us':
            st.write('This is an interactive web page for our ML project')
            
if __name__ == '__main__':
    main()