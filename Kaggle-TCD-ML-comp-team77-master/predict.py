import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import CategoricalDtype # 

# Input data files are available in the "./input/" directory.

import os
for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split     # Split labelled data into training and validation datasets
from sklearn.compose import make_column_transformer      # Combine outputs of multiple transformer objects applied to column subsets of the original feature space 
from sklearn.metrics import mean_absolute_error          # Mean squared error regression loss
from catboost import CatBoostRegressor                   # A fast, scalable, high performance Gradient Boosting on Decision Trees library
import lightgbm as lgb
from IPython.display import HTML


# Helper functions
def preprocess(df1, df2):
    train_df = df1.copy()
    test_df = df2.copy()

    # Dropping rows with nulls in YoR, Age and Profession
    train_df = train_df.drop(train_df.index[train_df['Year of Record'].isna()])
    train_df = train_df.drop(train_df.index[train_df['Age'].isna()])
    train_df = train_df.drop(train_df.index[train_df['Profession'].isna()])
    
    # Dropping unwanted columns
    train_df.drop(columns=['Instance','Wears Glasses','Crime Level in the City of Employement'],inplace=True)
    test_df.drop(columns=['Instance','Wears Glasses','Crime Level in the City of Employement'],inplace=True)

    # Taking log of income in labelled data to get normal distribution
    train_df['Total Yearly Income [EUR]'] = training_data['Total Yearly Income [EUR]'].apply(np.log)

    train_df.fillna(np.nan,inplace=True)
    test_df.fillna(np.nan,inplace=True)
    return train_df, test_df


def rename_and_lower(df):
    import re
    result = df.copy()
    for col in result.columns:
        column = re.sub(r' (\[|\().*', '', col.lower())
        column = re.sub(r' ', '_', column)
        result = result.rename(index=str, columns={col : column})
    return result.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)


def impute(df):
    result = df.copy()
    result['year_of_record'].fillna(result['year_of_record'].mean(),inplace=True)
    result['work_experience_in_current_job'].replace('#num!',np.nan, inplace=True)
    result['work_experience_in_current_job'] = result['work_experience_in_current_job'].astype('float64')
    result['work_experience_in_current_job'].fillna(result['work_experience_in_current_job'].mean(),inplace=True)
    result['satisfation_with_employer'].fillna('average', inplace=True)
    result["satisfation_with_employer"] = result["satisfation_with_employer"].astype(str)
    result['yearly_income_in_addition_to_salary'] = result['yearly_income_in_addition_to_salary'].astype("category")
    result['yearly_income_in_addition_to_salary'] = result['yearly_income_in_addition_to_salary'].str.replace(r' eur','').astype('float64')
    result['yearly_income_in_addition_to_salary'].fillna(result['yearly_income_in_addition_to_salary'].mean(),inplace=True)
    result['gender'].fillna('other', inplace=True)
    result['gender'].replace('unknown','other', inplace=True)
    result['gender'].replace('f','female', inplace=True)
    result['body_height'].fillna(result.groupby('gender')["body_height"].transform("mean"),inplace=True)
    result['hair_color'].fillna('other', inplace=True)
    result['profession'].fillna('other', inplace=True)
    result['university_degree'].fillna('no', inplace=True)
    result['country'].fillna('other', inplace=True)
    result['university_degree'].replace('0', 'no', inplace=True)
    result['housing_situation'] = result['housing_situation'].replace([0, '0', 'nA'], np.nan)
    result['housing_situation'].fillna('other', inplace=True)
    result['housing_situation'] = result['housing_situation'].astype(str)
    result['total_yearly_income'] = result['total_yearly_income'].fillna(result['total_yearly_income'].mean())
    result['total_yearly_income'] = result['total_yearly_income'].astype('float64')
    return result


def predictIncome(test_df,model):
    X = test_df.drop(columns=['total_yearly_income'])
    X = pd.get_dummies(X, prefix_sep='_', drop_first=True)
    y = test_df['total_yearly_income']

    j_test = model.predict(X)
    return np.exp(j_test)


def writeOutput(df, filename):
    output_file = pd.read_csv(filename)
    output_file['total_yearly_income'] = df
    output_file.to_csv(filename,index=False)


def create_download_link(title = "Download CSV file", filename = "Output1.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)


def main():

    # Loading Data
    training_data = pd.read_csv(r'../input/tcd-ml-comp-201920-income-pred-group/tcd-ml-1920-group-income-train.csv', sep=',', error_bad_lines=False, index_col=False, low_memory=False).drop_duplicates()
    predict_data = pd.read_csv(r'../input/tcd-ml-comp-201920-income-pred-group/tcd-ml-1920-group-income-test.csv', sep=',', error_bad_lines=False, index_col=False, low_memory=False)

    # Preprocessing     
    training_data, predict_data = preprocess(training_data, predict_data)


    # Renaming columns and making all categorical feature values to lowercase
    training_data = rename_and_lower(training_data)
    predict_data = rename_and_lower(predict_data)


    # Handeling null values
    training_data = impute(training_data)
    predict_data = impute(predict_data)

    y = training_data['total_yearly_income']

    # Combining Training and Test data sets to get all possible values of a categorical feature
    train_plus_test = pd.concat(objs=[training_data,predict_data], axis=0, sort=True)


    # Making non numeric columns as CategoricalDtype
    for column in train_plus_test.select_dtypes(include=[np.object]).columns:
        training_data[column] = training_data[column].astype(CategoricalDtype(categories = train_plus_test[column].unique()))
        predict_data[column] = predict_data[column].astype(CategoricalDtype(categories = train_plus_test[column].unique()))


    X = training_data.drop(columns=['total_yearly_income']) 

    # Split data into train and validate datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0)


    # One Hot Encode Categorical features
    X_train = pd.get_dummies(X_train, prefix_sep='_', drop_first=True)
    X_test = pd.get_dummies(X_test, prefix_sep='_', drop_first=True)


    # Initialize Model parameters
    #categorical_features_indices = np.where(x_train.dtypes != np.float)[0]
    model=CatBoostRegressor(
        iterations=7000, 
        depth=4, 
        learning_rate=0.03, 
        loss_function='MAE',
        verbose=1000, 
        od_type="Iter", 
        od_wait=500,
        use_best_model=True,
        task_type='GPU')

    # Train model with labelled dataset
    model.fit(X_train, y_train, eval_set=(X_test, y_test),plot=True)

    # Run rediction on validattion data split and check MAE
    j_validate = model.predict(X_test)
    print("Mean Absolute Error: ", mean_absolute_error(np.exp(y_test), np.exp(j_validate)))


    prediction = predictIncome(predict_data, model)

    output_file = '../input/tcd-ml-comp-201920-income-pred-group/tcd-ml-1920-group-income-submission.csv'

    # Write prediction to output file
    writeOutput(prediction, output_file)

    # create a link to download the dataframe which was saved with .to_csv method
    create_download_link(filename=output_file)


if __name__ == '__main__':
    main()
