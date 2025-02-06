import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import io
import base64
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import os
import prediction_system

# Load the dataset
dataset = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

# Function for Data Preprocessing
def preprocess_data(dataset):
    # Handle Missing Values
    st.subheader("Handling Missing Values")
    st.write("Select a strategy for handling missing values:")
    # Select strategy for continuous and categorical attributes
    cont_strategy = st.selectbox("Strategy for continuous attributes:", ["None","Mean", "Median"])
    cat_strategy = st.selectbox("Strategy for categorical attributes:", ["None" ,"Most Frequent", "Drop Rows"])
    if cont_strategy == "None":
        st.write("handling missing values are done for numerical data")
    else:
        # Continuous columns
        cont_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        imputer_cont = SimpleImputer(strategy=cont_strategy.lower())
        dataset[cont_columns] = imputer_cont.fit_transform(dataset[cont_columns])
    if cat_strategy == "None":
        st.write("handling missing values not done for categorical data")
    else:
        # Categorical columns
        cat_columns = dataset.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_strategy == "Most Frequent":
            imputer_cat = SimpleImputer(strategy="most_frequent")
            dataset[cat_columns] = imputer_cat.fit_transform(dataset[cat_columns])
        elif cat_strategy == "Drop Rows":
            dataset = dataset.dropna()
        st.success("Missing values handled.")
    
    st.write(dataset.head())
    
    # Encode Categorical Variables
    st.subheader("Encoding Categorical Variables")
    cat_columns = dataset.select_dtypes(exclude=[np.number]).columns.tolist()
    selected_cat_columns = st.multiselect("Select categorical columns to encode:", cat_columns)
    
    if len(selected_cat_columns) > 0:
        label_encoders = {}
        for col in selected_cat_columns:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])
            label_encoders[col] = le
        st.success("Selected categorical features encoded.")
    
    # Remove Irrelevant Features
    st.subheader("Removing Irrelevant Features")
    features_to_remove = st.multiselect("Select columns to remove:", dataset.columns.tolist())
    
    if len(features_to_remove) > 0:
        dataset.drop(columns=features_to_remove, inplace=True)
        st.success("Selected features removed.")
    
    return dataset

# Streamlit page setup
st.set_page_config(page_title="Heart Stroke Prediction", layout="wide")
st.sidebar.image("heartstroke.jpg", use_container_width=True)

# Navigation Sidebar
st.sidebar.title("Heart Stroke Prediction")
option = st.sidebar.selectbox("Select an option", ["Home", "Dataset Information", "Data Visualization","Data Preprocessing","Data Modeling","Prediction"])

if option == "Home":
    st.header("Welcome to the Heart Stroke Prediction App")
    st.write("Explore the dataset and visualize important aspects related to heart stroke prediction.")
    st.write(dataset)

elif option == "Dataset Information":
    st.header("Dataset Inspection")

    # Display Dataset Head, Tail, Info, Describe
    st.subheader("Dataset Head")
    st.write(dataset.head())
    
    st.subheader("Dataset Tail")
    st.write(dataset.tail())

    st.subheader("Info of the Dataset")
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Dataset Description")
    st.write(dataset.describe())

    st.subheader("Missing Values Information")
    st.write(dataset.isnull().sum())

    st.subheader("Duplicates values Information")
    st.write(dataset.duplicated().sum())

elif option == "Data Visualization":
    def render_mpl_fig(fig):
        buf=io.BytesIO()
        fig.savefig(buf,format="png",bbox_inches="tight")
        buf.seek(0)
        return buf
    
    st.title("Data Visualization for Heart Stroke Prediction")
    st.write("----------------------------------------------------------")

    st.subheader("Stroke Occurrence Distribution (Target Variable)")
    fig,ax=plt.subplots()
    sns.countplot(x='stroke',data=dataset,hue=None,palette='pastel',ax=ax)
    ax.set_title("Stroke Occurrence Count")
    ax.set_xlabel("Stroke")
    ax.set_ylabel("Count")
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.subheader("Age Distribution")
    fig,ax=plt.subplots()
    sns.histplot(dataset['age'],bins=20,kde=True,color='blue',ax=ax)
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.subheader("BMI Distribution")
    fig,ax=plt.subplots()
    sns.boxplot(x='stroke',y="bmi",data=dataset,ax=ax)
    ax.set_title("BMI Distribution by Stroke Occurrence")
    ax.set_xlabel("Stroke")
    ax.set_ylabel("BMI")
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.subheader("Smoking Status Distribution")
    fig,ax=plt.subplots()
    sns.countplot(x="smoking_status",data=dataset,palette='muted',hue=None,ax=ax)
    ax.set_title("Smoking Status Distribution")
    ax.set_xlabel("Smoking Status")
    ax.set_ylabel("Count")
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.subheader("Stroke Occurrence by Hypertension")
    fig,ax=plt.subplots()
    sns.barplot(x='hypertension',y='stroke',data=dataset,ci=None,errorbar=None,palette='viridis',ax=ax)
    ax.set_title("Stroke Occurrence by Hypertension")
    ax.set_xlabel("Hypertension")
    ax.set_ylabel("Average Stroke Occurrence")
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.subheader("Storke Occurrence by Age and Hypertension")
    fig,ax=plt.subplots()
    sns.scatterplot(x='age',y='stroke',hue='hypertension',data=dataset,palette='viridis',ax=ax)
    ax.set_title("Stroke Occurrence by Age and Hypertension")
    ax.set_xlabel("Age")
    ax.set_ylabel("Stroke")
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.subheader("Stroke Occurrence by Smoking Status and Stroke")
    fig,ax=plt.subplots()
    sns.barplot(x='smoking_status',y='stroke',data=dataset,errorbar=None,ci=None,palette='Set2',ax=ax)
    ax.set_title("Stroke Occurrence by Smoking Status")
    ax.set_xlabel("Smoking Status")
    ax.set_ylabel("Average Stroke Occurrence")
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.subheader("Pairplot for key Features")
    important_features=['age','bmi','avg_glucose_level','stroke']
    fig=sns.pairplot(dataset[important_features],hue='stroke',palette='husl')
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.subheader("Stroke Occurrence by Age and Glucose Levels")
    fig,ax=plt.subplots()
    sns.scatterplot(x='avg_glucose_level',y='age',hue='stroke',data=dataset,palette='coolwarm',ax=ax)
    ax.set_title("Stroke Occurrence by Age and Glucose Levels")
    ax.set_xlabel("Average Glucose Level")
    ax.set_ylabel("Age")
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.header("Handling Missing Data and Outliers")
    fig,ax=plt.subplots()
    sns.heatmap(dataset.isnull(),cbar=False,cmap='viridis',ax=ax)
    ax.set_title("Missing Values in Dataset")
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.header("Boxplot for BMI Outliers")
    fig,ax=plt.subplots()
    sns.boxplot(x=dataset['bmi'],ax=ax)
    ax.set_title("BMI Boxplot for Outliers")
    ax.set_xlabel("BMI")
    bug=render_mpl_fig(fig)
    st.image(bug)

    st.header("Class Imbalance")
    columns_for_class=dataset.columns.tolist()
    value=st.selectbox("choose the columns to check class imbalance",columns_for_class)
    if value:
        st.subheader(f"Class Distribution {value}")
        stroke_counts=dataset[value].value_counts()
        st.write(stroke_counts)
        
        st.subheader(f"Class Imbalance for {value}")
        fig,ax=plt.subplots()
        sns.countplot(x='stroke',data=dataset,hue=None,palette='pastel',ax=ax)
        ax.set_title(f"Class Imbalance for {value}")
        ax.set_xlabel(value)
        ax.set_ylabel("Count")
        bug=render_mpl_fig(fig)
        st.image(bug)

    # 7. Summarize Insights
    st.header("7. Summary of Insights")
    st.write("""
    - Age is positively correlated with stroke occurrence.
    - Hypertension and elevated glucose levels significantly increase stroke risk.
    - Smokers are more prone to strokes compared to non-smokers.
    - Class imbalance in the dataset should be addressed when building predictive models.
    """)


elif option == "Data Preprocessing":
    st.header("Data Preprocessing")
    
    # Show the original dataset before preprocessing
    st.subheader("Original Dataset")
    st.write(dataset.head())
    
    # Perform data preprocessing
    processed_data = preprocess_data(dataset)

    # Show the processed dataset
    st.subheader("Processed Dataset")
    st.write(processed_data.head())

    # Optionally, allow downloading the preprocessed dataset
    def save_and_download_data(processed_data, save_path="preprocessed_data.csv"):
        # Save the dataset to the specified path
        processed_data.to_csv(save_path, index=False)

        # Encode the CSV for downloading
        csv = processed_data.to_csv(index=False).encode('utf-8')

        # Streamlit download button
        st.download_button(
            label="Download Preprocessed Dataset as CSV",
            data=csv,
            file_name="preprocessed_data.csv",
            mime="text/csv",
            )

        return save_path
        # Assuming `processed_data` is your preprocessed DataFrame
    st.subheader("Download Preprocessed Dataset")

    # Specify the file path where you want to save the dataset
    save_path = "preprocess_data/preprocessed_data.csv"  # Replace with your desired file path

    # Call the function to save and allow downloading
    save_and_download_data(processed_data, save_path)

    st.write(f"The preprocessed dataset is saved at: `{os.path.abspath(save_path)}`")


elif option=="Data Modeling":
    
    st.title("Data Modeling")
    # Split data into features and target
    data=pd.read_csv("preprocess_data/preprocessed_data.csv")
    X = data.drop('stroke', axis=1)
    y = data['stroke']
    # Handle scaling for numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    # Function for SMOTE oversampling
    def oversampling(X_train, y_train):
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled,y_train_resampled

    # Model Selection
    model_choice = st.sidebar.selectbox(
        "Choose an Algorithm",
        ("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost","SVC","KNN","GradientBoostingClassifier")
    )    
    # Initialize the model
    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
    elif model_choice == "SVC":
        model = SVC()
    elif model_choice == "GradientBoostingClassifier":
        model = GradientBoostingClassifier()
    
    # Train the model
    if st.sidebar.button("Train Model"):
        with st.spinner("Training the model..."):
            X_train,y_train=oversampling(X_train,y_train)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
                }
            class_report = classification_report(y_test, y_pred, output_dict=True)
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)

            st.success("Model training complete!")
            # Display metrics
            st.subheader("Evaluation Metrics")
            for metric, value in metrics.items():
                st.write(f"{metric}: {value:.4f}")
                
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            def render_mpl_fig(fig):
                buf=io.BytesIO()
                fig.savefig(buf,format="png",bbox_inches="tight")
                buf.seek(0)
                return buf

            plt.figure(figsize=(8,6))
            sns.heatmap(
                cm,annot=True,fmt="d",cmap="Blues",xticklabels=["No Stroke","Stroke"],yticklabels=["No Stroke","Stroke"]
            )
            plt.title(f"Confusion Matrix ={model_choice}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            buf=render_mpl_fig(plt)
            st.image(buf)

            
            # Plot ROC curve
            st.subheader("ROC Curve")
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['ROC-AUC']:.4f})")
            plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_choice}")
            plt.legend(loc="lower right")
            buf=render_mpl_fig(plt)
            st.image(buf)


            # Create sigmoid graph for Logistic Regression (only for Logistic Regression)
            if model_choice == "Logistic Regression":
                st.subheader("Threshold graph")
                # Set a custom threshold (e.g., 0.5) and make predictions
                threshold = 0.5
                Y_pred_threshold = (y_proba >= threshold).astype(int)
                # Plotting the sigmoid function to visualize probabilities and threshold
                x_values = np.linspace(-5, 5, 100)
                sigmoid = 1 / (1 + np.exp(-x_values))
                plt.figure(figsize=(10, 6))
                plt.plot(x_values, sigmoid, label="Sigmoid Function", color="blue")
                plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
                plt.scatter(model.decision_function(X_test), y_proba, c=y_test, cmap='coolwarm', alpha=0.7, edgecolors='k')
                plt.xlabel("Model Decision Function")
                plt.ylabel("Predicted Probability")
                plt.title("Sigmoid Curve with Stroke Probabilities and Threshold")
                plt.legend()
                bug=render_mpl_fig(plt)
                st.image(bug)
            else:
                st.write(" ")
                
            # Feature importance bar graph 
            if model_choice == "Random Forest":
                st.subheader("Feature Importance in stroke prediction")
                feature_importance=model.feature_importances_
                sorted_idx=np.argsort(feature_importance)

                plt.figure(figsize=(8,5))
                plt.barh(X.columns[sorted_idx],feature_importance[sorted_idx],color="skyblue")
                plt.xlabel("Feature Importance")
                plt.ylabel("features")
                plt.title("Feature Importance in stroke prediction")
                bug=render_mpl_fig(plt)
                st.image(bug)
            elif model_choice == "XGBoost":
                st.subheader("Feature Importance in stroke prediction")
                feature_importance=model.feature_importances_
                sorted_idx=np.argsort(feature_importance)

                plt.figure(figsize=(8,5))
                plt.barh(X.columns[sorted_idx],feature_importance[sorted_idx],color="skyblue")
                plt.xlabel("Feature Importance")
                plt.ylabel("features")
                plt.title("Feature Importance in stroke prediction")
                bug=render_mpl_fig(plt)
                st.image(bug)
            else:
                st.write(" ")

elif option=="Prediction":
    dataset=pd.read_csv("data/healthcare-dataset-stroke-data.csv")
    prediction_system.stroke_prediction_app(dataset)
