import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


# Function to read CSV file and store data in a global variable
@st.cache
def load_data(file):
    return pd.read_csv(file)

st.title("Machine Learning Algorithm")
st.header("Analyze algorithms here: ")

# File Upload
upload_file = st.file_uploader("Upload Dataset", type=["csv"])

if upload_file is not None:
    # Load data when file is uploaded
    df = load_data(upload_file)

    if st.button("Glance"):
        st.dataframe(df.head(5))
    # Display dataset information
    if st.button("Details"):
        st.header("Dataset Information:")
        st.write(f"Number of Rows: {df.shape[0]}")
        st.write(f"Number of Columns: {df.shape[1]}")
        st.write("Column Names:")
        st.write(df.columns.tolist())
        st.write("Data Types:")
        st.write(df.dtypes)
        st.write("Memory Usage:")
        st.write(df.memory_usage().sum())

    tab_names = ["Decision Tree", "Logistic Regression", "XGBoost", "Random Forest"]

    selected_tab = st.radio("Select Algorithm:", tab_names)

    if selected_tab == "Decision Tree":
        # Decision Tree Algorithm
        st.header("Decision Tree Algorithm")
        if df is not None:
            # Feature columns
            feature_cols = ['Age', 'Smokes', 'AreaQ', 'Alkhol']

            # Features and target variable
            X = df[feature_cols] # Features
            y = df['Result'] # Target variable

            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

            # Create Decision Tree classifier object
            clf = DecisionTreeClassifier()

            # Train Decision Tree Classifier
            clf = clf.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = clf.predict(X_test)

            # Model Accuracy, how often is the classifier correct?
            accuracy = metrics.accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy}")

            # Compute confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

            # Display confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix)

             # SHAP Explanation
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)

            # SHAP Explanation
            st.header("SHAP Explanation:")
            if shap_values is not None:
                st.write("SHAP values shape:", shap_values.shape)
                st.write("X_test shape:", X_test.shape)

                fig, ax = plt.subplots(figsize=(10, 6))

                for i in range(X_test.shape[1]):
                    ax.scatter(X_test.iloc[:, i], shap_values[:, i, 0], label=f"Feature {i}", alpha=0.5)

                ax.set_xlabel("Feature Value")
                ax.set_ylabel("SHAP Value")
                ax.legend()

                st.pyplot(fig)
            else:
                st.write("SHAP values are not available.")

            # LIME Explanation
            explainer_lime = LimeTabularExplainer(X_train.values, mode="classification", feature_names=feature_cols)
            # Select an instance for explanation (you can change the index as needed)
            instance_index = 0
            instance = X_test.iloc[[instance_index]]
            exp = explainer_lime.explain_instance(instance.values[0], clf.predict_proba, num_features=len(feature_cols))

            st.header("LIME Explanation:")
            st.write(exp.as_pyplot_figure())

    elif selected_tab == "Logistic Regression":
        # Implement Logistic Regression Algorithm
        # Logistic Regression Algorithm
        st.header("Logistic Regression Algorithm")
        if df is not None:
            # Feature columns
            feature_cols = ['Age', 'Smokes', 'AreaQ', 'Alkhol']

            # Features and target variable
            X = df[feature_cols] # Features
            y = df['Result'] # Target variable

            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

            # Create Logistic Regression classifier object
            clf = LogisticRegression()

            # Train Logistic Regression Classifier
            clf = clf.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = clf.predict(X_test)

            # Model Accuracy, how often is the classifier correct?
            accuracy = metrics.accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy}")

            # Compute confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

            # Display confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix)

             # Create a LinearExplainer for logistic regression
            # Compute SHAP values
            # SHAP Explanation
            explainer = shap.LinearExplainer(clf, X_train)
            shap_values = explainer.shap_values(X_test)

            # SHAP Explanation
            st.header("SHAP Explanation:")
            if shap_values is not None:
                st.write("SHAP values shape:", shap_values.shape)
                st.write("X_test shape:", X_test.shape)

                fig, ax = plt.subplots(figsize=(10, 6))

                for i in range(X_test.shape[1]):
                    ax.scatter(X_test.iloc[:, i], shap_values[:, i], label=f"Feature {i}", alpha=0.5)

                ax.set_xlabel("Feature Value")
                ax.set_ylabel("SHAP Value")
                ax.legend()

                st.pyplot(fig)
            else:
                st.write("SHAP values are not available.")

            # LIME Explanation
            explainer_lime = LimeTabularExplainer(X_train.values, mode="classification", feature_names=feature_cols)
            instance_index = 0  # Select an instance for explanation (you can change the index as needed)
            instance = X_test.iloc[[instance_index]]
            exp = explainer_lime.explain_instance(instance.values[0], clf.predict_proba, num_features=len(feature_cols))

            st.header("LIME Explanation:")
            st.write(exp.as_pyplot_figure())
        pass


    elif selected_tab == "XGBoost":
        # Implement XGBoost Algorithm
         # XGBoost Algorithm
        st.header("XGBoost Algorithm")
        if df is not None:
            # Feature columns
            feature_cols = ['Age', 'Smokes', 'AreaQ', 'Alkhol']

            # Features and target variable
            X = df[feature_cols] # Features
            y = df['Result'] # Target variable

            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

            # Create XGBoost classifier object
            clf = XGBClassifier()

            # Train XGBoost Classifier
            clf = clf.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = clf.predict(X_test)

            # Model Accuracy, how often is the classifier correct?
            accuracy = metrics.accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy}")

            # Compute confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

            # Display confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix)

            # SHAP Explanation
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)

            # SHAP Explanation
            st.header("SHAP Explanation:")
            if shap_values is not None:
                st.write("SHAP values shape:", shap_values.shape)
                st.write("X_test shape:", X_test.shape)

                fig, ax = plt.subplots(figsize=(10, 6))

                for i in range(X_test.shape[1]):
                    ax.scatter(X_test.iloc[:, i], shap_values[:, i], label=f"Feature {i}", alpha=0.5)  # Removed the third index here

                ax.set_xlabel("Feature Value")
                ax.set_ylabel("SHAP Value")
                ax.legend()

                st.pyplot(fig)
            else:
                st.write("SHAP values are not available.")
            # LIME Explanation
            st.header("LIME Explanation:")
            explainer_lime = LimeTabularExplainer(X_train.values, mode="classification", feature_names=X_train.columns)
            instance_index = 0  # Select an instance for explanation (you can change the index as needed)
            instance = X_test.iloc[[instance_index]]
            exp = explainer_lime.explain_instance(instance.values[0], clf.predict_proba, num_features=len(X_train.columns))
            st.write(exp.as_pyplot_figure())

        pass
    elif selected_tab == "Random Forest":
        # Implement Random Forest Algorithm
         # Random Forest Algorithm
        st.header("Random Forest Algorithm")
        if df is not None:
            # Feature columns
            feature_cols = ['Age', 'Smokes', 'AreaQ', 'Alkhol']

            # Features and target variable
            X = df[feature_cols] # Features
            y = df['Result'] # Target variable

            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

            # Create Random Forest classifier object
            clf = RandomForestClassifier()

            # Train Random Forest Classifier
            clf = clf.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = clf.predict(X_test)

            # Model Accuracy, how often is the classifier correct?
            accuracy = metrics.accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy}")

            # Compute confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

            # Display confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix)

        # SHAP Explanation
        explainer_shap = shap.TreeExplainer(clf)
        shap_values = explainer_shap.shap_values(X_test)

        # SHAP Explanation
        st.header("SHAP Explanation:")
        if shap_values is not None:
            st.write("SHAP values shape:", shap_values.shape)
            st.write("X_test shape:", X_test.shape)

            fig, ax = plt.subplots(figsize=(10, 6))

            for i in range(X_test.shape[1]):
                ax.scatter(X_test.iloc[:, i], shap_values[:, i, 0], label=f"Feature {i}", alpha=0.5)

            ax.set_xlabel("Feature Value")
            ax.set_ylabel("SHAP Value")
            ax.legend()

            st.pyplot(fig)
        else:
            st.write("SHAP values are not available.")


        # LIME Explanation
        st.header("LIME Explanation:")
        explainer_lime = LimeTabularExplainer(X_train.values, mode="classification", feature_names=X_train.columns)
        instance_index = 0  # Select an instance for explanation (you can change the index as needed)
        instance = X_test.iloc[[instance_index]]
        exp = explainer_lime.explain_instance(instance.values[0], clf.predict_proba, num_features=len(X_train.columns))
        st.write(exp.as_pyplot_figure())
        pass
