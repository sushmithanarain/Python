

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns  #plot
#statsmodel
import statsmodels.api as sm
#SCIKIT-LEARN
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression


# In[2]:


data=pd.read_csv("", index_col="")
data.head()







data.describe()



data.isnull()



data=data.dropna()
data.head()



data.isnull().any()



data['Genderd']=data['Gender'].replace({'Male':0,'Female':1})
data=data.drop('Gender',axis=1)
data.head()
#axis=1 is columns and axis=0 is row


# In[10]:


sns.pairplot(data)


# In[11]:


data.hist()


# In[12]:


y=data['Y']
X=data[['Xd','X2','X3']]


# In[13]:


X=sm.add_constant(X)
log_reg=sm.Logit(y,X, data=data).fit()
log_reg.summary()
#print(log_reg.summary())
#we are looking athe log odds, the value of intercept and other will be converted in to the probability values which will fall in between 0 and 1; 
#in logistic regression we dont interpret the coefficient the way be do in linear regression. 


# In[14]:


odds_ratios=pd.DataFrame(
{
    "OR":log_reg.params,
    "Lower CI":log_reg.conf_int()[0],
    "Upper CI":log_reg.conf_int()[1],
}
)
odds_ratios=np.exp(odds_ratios)
print(odds_ratios)


# In[15]:


from sklearn.metrics import classification_report,confusion_matrix


# In[16]:


X=data.iloc[:,[0,1,3]]
y=data.iloc[:,2]
#[row:,columns]
X.head()




#feature scaling to normalise the range of independent 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
X



#splitting the data into test and train
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=42)





model1=LogisticRegression()
model1.fit(X_train,y_train)


# In[20]:


model1.intercept_


# In[21]:


y_pred=model1.predict(X_test)
results=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
results


# In[22]:


plt.scatter(y_test,y_pred)


# In[23]:


model1.coef_


# In[24]:


model1.predict_proba(X)


# In[25]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


cnf_matrix=confusion_matrix(y_test, y_pred)
cnf_matrix



sns.heatmap(pd.DataFrame(cnf_matrix), cmap="coolwarm", annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')


# In[28]:


#accuracy
#TN+TP/(TN+TP+FP+FN)
accuracy=accuracy_score(y_test,y_pred)
accuracy


# In[29]:


#Precision
#out of all the positive predicted, what percentage is truly positive
#precision=TP/(TP+FP)
precision=precision_score(y_test,y_pred)
precision


# In[30]:


#recall
#out of the total postive, what percentage are predicted positive
#recall=TP/(TP+FN)
recall=recall_score(y_test,y_pred)
recall




#f1 score
#a harmonic mean of the precison and recalll'#best value 1 , worst 0
#F1=2*(precision*recall)/(precison+recall)
f1=f1_score(y_test,y_pred)
f1


# In[32]:


#panada.create_dummies

