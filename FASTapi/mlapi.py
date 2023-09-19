# Bring in lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR

app = FastAPI()

class ScoringItem(BaseModel):
    company_name: str
    Rating: float
    Size:  str
    TypeOfOwnership: str
    Industry: str
    Sector: str
    Revenue: str
    job_title: str
    job_state: str
    age: int
    tools: int
    techs: int
    education: int
    seniority: str
    desc_len: int
    hourly: int
    employer_provided: int

with open('svm_model_0.1.0.pkl', 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['svm_model']

DATA_PATH = '../Data/Processed/0_DataCleaned_df.csv'

df_model = pd.read_csv(DATA_PATH)
df_model = df_model[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','job_title',
            'job_state','age','tools','techs','education','seniority','desc_len','company_name','hourly','employer_provided']]

num_cols = df_model[['Rating','age','tools','techs','education','desc_len','hourly','employer_provided']].reset_index(drop = True)
cat_cols = df_model.drop(columns=['avg_salary','Rating','age','tools','techs','education','desc_len',
                                        'hourly','employer_provided'], axis =1).reset_index(drop=True)

df_model_null = df_model.copy()
df_model_null.replace(to_replace={-1: np.NAN,'-1':np.NAN},inplace=True) 
df_dummy_null = pd.get_dummies(df_model_null,dummy_na=True)


df_dummy_null['avg_salaries_cat'] = pd.cut(df_dummy_null['avg_salary'], bins=[0.,75,100,135,170,np.inf],
                                    labels=[0,1,2,3,4])
split = StratifiedShuffleSplit(n_splits =1, test_size =0.2, random_state=42)
for train_index, test_index in split.split(df_dummy_null,df_dummy_null['avg_salaries_cat']):
    strat_train_set_dum_null = df_dummy_null.loc[train_index]
    strat_test_set_dum_null = df_dummy_null.loc[test_index]

for strat_set in (strat_train_set_dum_null,strat_test_set_dum_null,df_dummy_null):
    strat_set.drop("avg_salaries_cat",axis = 1, inplace=True)

x_train_dum_null = strat_train_set_dum_null.drop("avg_salary",axis=1)
y_train_dum_null = strat_train_set_dum_null['avg_salary'].copy()

num_attribs_dum_null = list(num_cols.columns)
cat_attribs_dum_null = list(x_train_dum_null.iloc[:,8:])


svm_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('min_max_scaler', MinMaxScaler()),
])

x_train_svm_prepared = svm_pipeline.fit_transform(x_train_dum_null)

def pipeline(df):
    df.replace(to_replace={-1: np.NAN,'-1':np.NAN},inplace=True) 
    df = pd.get_dummies(df,dummy_na=True)
    new_df = pd.DataFrame(data=df,columns=num_attribs_dum_null+cat_attribs_dum_null)
    new_df_transformed = svm_pipeline.transform(new_df)
    
    return new_df_transformed

@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    df_transformed = pipeline(df)
    yhat= model.predict(df_transformed)
    return {"prediction" : str(int(yhat))+"k peryear"}

