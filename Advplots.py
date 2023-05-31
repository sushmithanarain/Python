

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup


# In[17]:


text=("sexuality, spectrum, idea, people, sexual orientation,homosexual, gay, lesbian, queer, heterosexual, man, woman, homosexual, heterosexual,bisexual, sexuality,  sexuality,  spectrum, fluidity, sexual orientation,spectrum, sexual orientation, fixed orientations, sexuality, sexuality spectrum,pansexual, generational, Millennials, spectrum,rights, spectrum,millennial, millennials, Sexual Orientation, Spectrum,spectrum, singular entity, homosexuality, heterosexuality, LGBTQ, gay, lesbians, pansexual, bisexual, queer, love, abuse, abuse, violence, rights")
#display the generated image
wordcloud=WordCloud(width=2044, height=1600, background_color="whitesmoke").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0.5,y=0.5)
plt.show()


# In[4]:


import numpy as np
text = "Plots"

x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)


wc = WordCloud(background_color="white", repeat=True, mask=mask)
wc.generate(text)

plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.show()


# In[18]:


#get the HTML of the webpage
url="https://en.wikipedia.org/wiki/List_of_Indian_male_film_actors"
response=requests.get(url)
Call= BeautifulSoup(response.content, "html.parser")
#get the text from the webpage
text=Call.get_text()

#create wordcloud
wc=WordCloud(background_color="black", colormap="Blues")
wc.generate(text)
plt.imshow(wc)
plt.axis("off")
plt.show()
wc.to_file("CMBYN wordcloud.png")


# In[6]:


url="https://en.wikipedia.org/wiki/Jaun_Elia"
response=requests.get(url)
Jaun= BeautifulSoup(response.content, "html.parser")
#get the text from the webpage
text=Jaun.get_text()

#create wordcloud
wc=WordCloud(background_color="black", colormap="Blues")
wc.generate(text)
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[19]:


import pandas as pd
import plotly.express as px
import plotly
import json
import requests


# In[20]:


data=pd.read_csv("")
data


# In[9]:


fig=px.choropleth(data, 
                 geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
                 featureidkey='properties.ST_NM',
                 locations='state',
                 color='sdg score',
                 color_continuous_scale='inferno')
fig.update_geos(fitbounds="locations", visible=False)
fig.show()
fig.write_image("image_india.jpeg")


# In[21]:


df=pd.read_excel("")
df


# In[11]:


fig = px.sunburst(df, path=['State', 'City'], values='Population', color='Literacy', color_continuous_scale='magma')
fig.show()


# In[12]:


#bubble chart
fig1 = px.scatter(df, x="City", y="Sex Ratio",size="Population", color="State", size_max=60)
fig1.show()


# In[25]:


fig2 = px.treemap(df, path=['State', 'City'], values='Population', color='Literacy', color_continuous_scale='blues')
fig2.show()

