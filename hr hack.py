#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv("train_LZdllcl.csv")


# In[3]:


test = pd.read_csv("test_2umaH9m.csv")


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


train.shape, test.shape


# In[6]:


train.dtypes


# In[7]:


train.info()


# In[8]:


train.head(2)


# In[9]:


train.isna().sum()


# In[10]:


train.duplicated().sum()


# In[11]:


train.is_promoted.value_counts()


# In[12]:


train.is_promoted.value_counts().plot(kind='bar')


# In[13]:


tgt_col = ['is_promoted'] 
ign_cols = ['employee_id']


# In[14]:


train.describe().T


# In[15]:


train.drop(columns=ign_cols).describe().T


# In[16]:


train.describe(include='object').T


# In[17]:


train.nunique()


# In[18]:


for col in train.drop(columns=ign_cols).columns:
    print(col,train[col].nunique(),  '=>', train[col].unique())


# In[19]:


sns.distplot(train.avg_training_score)


# In[20]:


for col in train.select_dtypes(include='object').columns:
    plt.figure(figsize=(5,3))
    sns.countplot(y=train[col])
    plt.show()


# In[21]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc


# In[22]:


from sklearn import set_config


# In[23]:


set_config(display='diagram')


# In[24]:


train.dtypes


# In[25]:


cat_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel']


# In[26]:


num_cols = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 
            'KPIs_met >80%', 'awards_won?', 'avg_training_score']


# In[27]:


print(tgt_col, ign_cols, cat_cols, num_cols, sep='\n')


# In[28]:


cat_pipe_encode = Pipeline(
    steps=[
        ('impute_cat', SimpleImputer(strategy='most_frequent')),  
        ('ohe', OneHotEncoder(handle_unknown='ignore'))          
    ]
)


# In[29]:


num_pipe_encode = Pipeline(
steps = [
    ('impute_num', SimpleImputer(strategy='median')), 
    ('scale',StandardScaler())
])


# In[30]:


preprocess = ColumnTransformer(
    transformers =[
        ('cat_encode',cat_pipe_encode,cat_cols),
        ('num_encode',num_pipe_encode,num_cols)
    ]
)


# In[31]:


model_pipeline = Pipeline(
steps=[
    ('preprocess',preprocess),
    ('model',LogisticRegression())
])


# In[32]:


X = train.drop(columns=ign_cols+tgt_col)
X.head(2)


# In[33]:


y = train[tgt_col]
y.head(2)


# In[34]:


from sklearn.model_selection import train_test_split, GridSearchCV


# In[35]:


train_X, val_X, train_y, val_y = train_test_split(X,y, 
                                         random_state=42, test_size=0.1)
train_X.shape, val_X.shape, train_y.shape, val_y.shape


# In[36]:


model_pipeline


# In[37]:


model_pipeline.fit(train_X, train_y)


# In[38]:


model_pipeline.predict_proba(val_X)


# In[39]:


model_pipeline.predict_proba(val_X)[:,0]


# In[40]:


model_pipeline.predict_proba(val_X)[:,1]


# In[41]:


model_pipeline.predict(val_X)


# In[42]:


def model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline):
    
    predicted_train_tgt = model_pipeline.predict(train_X)
    predicted_val_tgt = model_pipeline.predict(val_X)

    print('Train AUC', roc_auc_score(train_y,predicted_train_tgt),sep='\n')
    print('Valid AUC', roc_auc_score(val_y,predicted_val_tgt),sep='\n')

    print('Train cnf_matrix', confusion_matrix(train_y,predicted_train_tgt),sep='\n')
    print('Valid cnf_matrix', confusion_matrix(val_y,predicted_val_tgt),sep='\n')

    print('Train cls_rep', classification_report(train_y,predicted_train_tgt),sep='\n')
    print('Valid cls rep', classification_report(val_y,predicted_val_tgt),sep='\n')

    # plot roc-auc
    y_pred_proba = model_pipeline.predict_proba(val_X)[:,1]
    plt.figure()
    fpr, tpr, thrsh = roc_curve(val_y,y_pred_proba)
    #roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr)
    plt.show()


# In[43]:


model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline)


# In[44]:


params = [
    {
    'model': [LogisticRegression()],
    'model__penalty':['l2',None],
    'model__C':[0.5,3]
    }    
]


# In[45]:


grid = GridSearchCV(estimator=model_pipeline, param_grid=params, 
                    cv=2, scoring='roc_auc')


# In[46]:


grid.fit(train_X, train_y)


# In[47]:


grid.best_params_


# In[48]:


res_df = pd.DataFrame(grid.cv_results_,)
pd.set_option('display.max_colwidth',100)
res_df[['params','mean_test_score','rank_test_score']]


# In[49]:


sub = pd.read_csv('sample_submission_M0L0uXE.csv')
sub.head(3)


# In[50]:


test.head(3)


# In[51]:


train.columns.difference(test.columns)


# In[52]:


sub['is_promoted'] = model_pipeline.predict(test)


# In[53]:


sub.to_csv('sub_1.csv',index=False)


# In[54]:


sub


# In[55]:


import joblib


# In[78]:


joblib.dump(model_pipeline,'jobchag_pipeline_model.pkl')


# In[79]:


from imblearn.over_sampling import RandomOverSampler


# In[80]:


over_sampling = RandomOverSampler()


# In[81]:


train_y.value_counts()


# In[82]:


train_X_os, train_y_os = over_sampling.fit_resample(train_X,train_y)


# In[83]:


train_y_os.value_counts()


# In[84]:


from sklearn.tree import DecisionTreeClassifier


# In[85]:


params_2 = [
    {
    'model': [LogisticRegression()],
    'model__penalty':['l2',None],
    'model__C':[0.5,3]
    },
    {
    'model': [DecisionTreeClassifier()],
    'model__max_depth':[3,5]
    }
]


# In[86]:


params_2


# In[87]:


grid_2 = GridSearchCV(estimator=model_pipeline, param_grid=params_2, 
                    cv=2, scoring='roc_auc')


# In[88]:


grid_2.fit(train_X_os, train_y_os)


# In[89]:


grid_2


# In[90]:


grid_2.best_params_


# In[91]:


grid_2.best_estimator_


# In[92]:


grid_2.cv_results_


# In[93]:


new_model = grid_2.best_estimator_


# In[94]:


model_train_val_eval(train_X,val_X,train_y,val_y,new_model)


# In[95]:


model_train_val_eval(train_X_os,val_X,train_y_os,val_y,new_model)


# In[96]:


res_df_2 = pd.DataFrame(grid_2.cv_results_,)
pd.set_option('display.max_colwidth',100)
res_df_2[['params','mean_test_score','rank_test_score']]


# In[97]:


sub['target'] = new_model.predict(test)
sub.to_csv('sub_2.csv',index=False)


# In[98]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier, StackingClassifier


# In[ ]:




