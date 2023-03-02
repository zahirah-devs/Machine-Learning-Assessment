import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from dataprep.eda import create_report
import warnings
warnings.filterwarnings('ignore')


#Read the data and put it in a dataframe
mk_data = pd.read_csv('data/bank_marketing_data.csv', sep=';')

# A Method that uses an eda package to visualise the raw data
def generate_eda_report(data):
    with warnings.catch_warnings(record=True):
        create_report(data).show_browser()
    
    
    
# Create age ranges and one-hot encode the column
age_bins = [18, 30, 40, 50, 60, np.inf]
age_labels = ['18-30', '30-40', '40-50', '50-60', '60+']
age_encoded_columns = pd.get_dummies(pd.cut(mk_data['age'], bins=age_bins, labels=age_labels), prefix='age')

# Combine the new age columns with the original data
mk_data = pd.concat([mk_data.drop('age', axis=1), age_encoded_columns], axis=1)

# One-hot encode the categorical variables
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
encoded_columns = pd.get_dummies(mk_data[categorical_columns], prefix=categorical_columns)

# Combine the new encoded columns with the original data
mk_data = pd.concat([mk_data.drop(categorical_columns, axis=1), encoded_columns], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mk_data.drop('target', axis=1), 
                                                    mk_data['target'], 
                                                    test_size=0.2, 
                                                    random_state=42)


#Remove NAN or large values and infinity values with 0
X_train = X_train.replace([np.inf, -np.inf, np.nan], 0)
y_train = y_train.replace([np.inf, -np.inf, np.nan], 0)
X_test = X_test.replace([np.inf, -np.inf, np.nan], 0)
y_test = y_test.replace([np.inf, -np.inf, np.nan], 0)


# Define the models
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC()
}

# Fit the models and generate the confusion matrices
cms = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    cms[name] = confusion_matrix(y_test, preds)
    

# Plot the confusion matrices
def plot_confusion_matrices(cms):
    with warnings.catch_warnings(record=True):
        fig, axes = plt.subplots(nrows=1, ncols=len(cms), figsize=(12, 4))
        labels = ['Not success', 'Success']

        for i, (name, cm, ax) in enumerate(zip(cms.keys(), cms.values(), axes.flatten())):
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
            for x in range(cm.shape[0]):
                for y in range(cm.shape[1]):
                    ax.text(y, x, s=cm[x, y], va='center', ha='center')
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_title(name)

        plt.show()

# Call the functions
def results_of_data():
    with warnings.catch_warnings(record=True):
        plot_confusion_matrices(cms)
    

