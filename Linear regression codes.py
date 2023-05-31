#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Setting working directory
os.getcwd()
os.chdir("")
os.getcwd()


# In[2]:


# Linear Regression


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  #plot
#statsmodel
import statsmodels.api as sm
#SCIKIT-LEARN
from sklearn import linear_model


# In[7]:


df=pd.read_csv()
df.head()


# In[8]:


df.describe()


# In[9]:


df.columns


# In[11]:


df.hist()


# In[12]:


#pairplot
sns.pairplot(df)


# In[16]:


# Correlation
correlations=df.corr()
#Heat maps
sns.heatmap(correlations, annot=True).set(title='')


# In[17]:


df.plot(x='a',y='b', style='o')
plt.title('- ')
plt.xlabel('a)
plt.ylabel('b')
plt.show()


# In[18]:


# Assigning dependent and independent variables
X =df[["x1","x2","x3"]]
y=df["Y"]
X.head(2)


# In[19]:


model=sm.OLS(y,X).fit()
model.summary()


# In[20]:


#add constant
X=sm.add_constant(X)
model=sm.OLS(y,X).fit()
model.summary()


# In[21]:


sns.regplot(x='x1',y="Y",data=df)
sns.lmplot(x='x2',y="Y",data=df)
sns.lmplot(x='x3',y="Y",data=df)


# In[22]:


#residual plot
#are residuals noramlly distributed?
sns.histplot(model.resid)


# In[23]:


#how much they are deviating from the line(fitted values), if residuals are largely deviated that means it is not a good fit
plt.scatter(model.fittedvalues, model.resid)
plt.xlabel('fitted values')
plt.ylabel('residuals')
plt.title('residuals vs fitted values')
plt.show()


# In[24]:


actual_fitted=pd.DataFrame({'Actual':y,'Fitted':model.predict(X)})
actual_fitted


# In[25]:


#Heteroscadisity
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test=het_breuschpagan(model.resid,X)
bp_test
p_value=bp_test[1]
p_value
#H0 (null hypothesis): data is homoscedastic. Ha (alternative hypothesis): data is heteroscedastic. 


# In[26]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif["Feature"]=X.columns
vif["VIF"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif
#multicollinearity


# In[27]:


#autocorrealtion
from statsmodels.stats.stattools import durbin_watson
dw_statistic=durbin_watson(model.resid)
dw_statistic
#dw<2 no autocorelation


# In[28]:


#normality
from statsmodels.stats.stattools import jarque_bera
jb_test=jarque_bera(model.resid)
jb_test[1]


# In[29]:


from stargazer.stargazer import Stargazer
from IPython.core.display import HTML


# In[30]:


#ols 
stargazer=Stargazer([model])
HTML(stargazer.render_html())


# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[33]:


y=df['Petrol_Consumption']
X=df[['Average_income', 'Population_Driver_licence(%)','Paved_Highways']]


# In[39]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=42)
#training dataset is 70% and testing dataset is 30%


# In[40]:


model1=LinearRegression()
model1.fit(X_train,y_train)


# In[41]:


model1.intercept_


# In[42]:


model1.coef_


# In[43]:


coefficients_df=pd.DataFrame(data=model1.coef_,index=X.columns,columns=['Coefficient value'])
coefficients_df


# In[ ]:


#checking if our model is making mistakes
#using the regression equation from training dataset to predict the test dataset
#using the training datasets we predict the y values, if they are equal to actual y values then it is good model. 


# In[44]:


y_pred=model1.predict(X_test)
results=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
results


# In[45]:


plt.scatter(y_test,y_pred)


# In[46]:


plt.hist(y_test-y_pred)


# In[53]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[51]:


mae1 = mean_absolute_error(y_test,y_pred)
mae1
# A higher MAE indicates that your predictions are less accurate or have a larger average discrepancy from the true values.


# In[54]:


mse=mean_squared_error(y_test,y_pred)
mse
#A higher MSE indicates that your predictions have a larger average squared discrepancy from the true values.


# In[55]:


rmse=np.sqrt(mse)
rmse

