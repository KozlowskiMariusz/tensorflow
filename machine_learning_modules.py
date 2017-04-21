# Module 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk


#Module 2 Numpy

import numpy as np

a = [2,3,4]
a1 = np.array(a)
b = [4,3,2]
b1 = np.array(b)
Differences between Numpy and Python List
print(a+b)
print(a1+b1)
print(a-b)
print(a1-b1)

a = [2, 'hi']
print(a)
a1 = np.array(a)
print(a1)

a = [2,3,5]
a1 = np.array(a,dtype=np.int16)
print(a1)
b1 = a1.astype(np.float64)
print(b1)
print(a1.dtype)
print(b1.dtype)

a = [[1,5,2],[4,5,6]]
b = [[6,5,4],[2,5,1]]
print(a+b)
a1 = np.array(a)
b1 = np.array(b)
print(a1+b1)

a = [[1,5,2],[4,5,6]]
a1 = np.array(a)
print(a1.ndim)
print(a1.shape)
print(len(a1))
print(a1.dtype)

a = np.linspace(1,10,40) 
a = np.logspace(-10,-1,10)
a = np.arange(1,10,2)
print(a)

reshape
a = np.arange(1,12,2)
print(a)
b = a.reshape(3,2)
print(b)
c = b.ravel()
print(c)
c = a.reshape(-1,2)
print(c)

stacking arrays
a = np.array([[1,1],[2,2],[3,3]])
b = np.array([[4,4],[5,5],[6,6]])
print(np.vstack([a,b]))
print(np.hstack([a,b]))

a = [[1,5,2],[4,5,6],[7,8,9]]
a1 = np.array(a)
print(a1)
print(a1[[0,2],:][:,[0,2]])

a = [3,4,2,8,6]
a1 = np.array(a)
print(a1[a1>3])

a = [3,4,7,-1,-2,6,8,-9,3]
a1 = np.array(a)
print(sum(a1[a1>0]))

Numpy Math Functions
print(np.exp(1))
print(np.sin(np.pi/2))
print(np.cos(np.pi/2))
print(np.sqrt(4))

Numpy Statistical Functions
a = [3,4,2,8,6]
a = [[1,5,2],[4,5,6],[7,8,9]]
a1 = np.array(a)
print(a1)
print(np.mean(a1,axis=0))
print(np.std(a1))

Random Number
np.random.seed(10)
print(np.random.rand(5))
print(np.random.randn(5))
print(np.random.randint(1,6,2))

Linear Algebra
a = np.array([[1,1],[1,1]])
b = np.array([[2,2],[2,2]])
print(a*b)
a = np.matrix([[1,1],[1,1]])
b = np.matrix([[2,2],[2,2]])
print(a*b)

from numpy import linalg
a = np.matrix([[2,4],[3,-1]])
print(a)
b = linalg.inv(a)
print(b)
print(a*b)






#Module 3 Matplolib

import numpy as np 
import matplotlib.pyplot as plt

x = np.linspace(-4,4,100)
y = np.sin(x)
y2 = np.cos(x)
y3 = y*y2
y4 = y*y -y2*y2
plt.plot(x,y,color='#334411',marker='o',linestyle='-')
plt.plot(x,y,'ro-',label='sine',x,y2,'g^-',label='cosine')
plt.subplot(2,1,1)
plt.plot(x,y,'ro-',label='sine')
plt.subplot(2,1,2)
plt.plot(x,y2,'g^-',label='cosine')
plt.grid()
plt.legend(loc='upperleft')
plt.legend(bbox_to_anchor=(1.1,1.05))
plt.xlabel('x')
plt.ylabel('y')
plt.title('sine curve')
plt.show()

Challenge
plt.subplot(2,2,1)
plt.plot(x,y,'ro')
plt.subplot(2,2,2)
plt.plot(x,y2,'g^')
plt.subplot(2,2,3)
plt.plot(x,y3,'b^')
plt.subplot(2,2,4)
plt.plot(x,y4,'ko')
plt.show()

Other Plots

Scatter Plot
x = np.linspace(0,10,200)
y = x + np.random.randn(len(x))
plt.scatter(x,y)
plt.show()

Bar Plot
Horizontal Bar Plot
people = ['Tom', 'Dick', 'Harry', 'Slim', 'Jim']
height = 170 + 20 * np.random.randn(len(people))
x = np.arange(len(people))
plt.barh(x,height,align="center",color='yellow')
plt.yticks(x,people)
plt.show()

Histogram
x = np.random.randn(100000)
plt.hist(x,10)
plt.show()

Contour Plot
x = np.linspace(-1,1,255)
y = np.linspace(-2,2,300)
X,Y = np.meshgrid(x,y)
z = np.sin(X)*np.cos(Y)
plt.contour(X,Y,z,10)
plt.show()

Pie Plot
x = [45,50,20]
plt.pie(x)
plt.show()







##Module 4 Pandas

import numpy as np
import pandas as pd

# Series

a = [2,5,6,7,3]
b = [3,4,7,9,10]
a1 = np.array(a)
b1 = np.array(b)
a2 = pd.Series(a,index=['a','b','c','d','e'])
b2 = pd.Series(b,index=['a','b','c','d','e'])
print(a1+b1)
# print(a2)
# print(b2)
# print(a2+b2)
print(a2['c'])

print(a2.index)
print(a2.values)

a = pd.Series(np.random.randn(1000))
print(a.head())
print(a.tail(10))
print(a[300:305])

Data Frame

a = [[3,4],[5,6]]
b = [[6,5],[4,3]]
a1 = np.array(a)
b1 = np.array(b)
print(a1+b1)

a2 = pd.DataFrame(a, index=[1,2], columns=['a','b'])
b2 = pd.DataFrame(b, index=[1,2], columns=['a','c'])
print(a2+b2)

a = pd.DataFrame(
    {
    'name': ['Ally','Jane','Belinda'],
    'height':[160,155,163],
    'age': [40,35,42]
    },
    columns = ['name','height','age'],
    index = ['101','105','108']
    )
print(a)
print(a.index)
print(a.columns)
print(a.values)

Column data
print(a['name'])
print(a.name)
print(a[[0]])

Row data
print(a.ix['105'])
print(a.ix[1])
print(a.loc['105'])

Scalar data
print(a.ix[1]['height'])
print(a.ix['105','height'])

a.index = ['108','105','110']
print(a)

Re-index
a3 = a.reindex(['108','105','110'],fill_value='NA')
a3 = a.reindex(['108','105','110'])
print(a3)
a4 = a3.dropna()
print(a4)
a = pd.DataFrame(np.random.randn(20,5))
print(a.head())
print(a.tail())


sp500 = pd.read_csv('data/sp500.csv',index_col='Symbol',usecols=[0,2,3,7])
print(sp500.head())
print(sp500[sp500.Price<100])

print(sp500[(sp500.Price>100) & (sp500.Sector=='Health Care')])


a = pd.DataFrame(
    {
    'name': ['Ally','Jane','Belinda'],
    'height':[160,155,163],
    'age': [40,35,42]
    },
    columns = ['name','height','age']
    )

b = pd.DataFrame(
    {
    'name': ['Ally','Jane','Alfred'],
    'weight': [55,50,80]
    },
    columns = ['name','weight']
    )

print(a)
print(b)

Inner Join
c = pd.merge(a,b,on='name',how='inner')
print(c)

Left Join
c = pd.merge(a,b,on='name',how='left')
print(c)

Right Join
c = pd.merge(a,b,on='name',how='right')
print(c)

Outer Join
c = pd.merge(a,b,on='name',how='outer')
print(c)

a = pd.DataFrame(
    {
    'name': ['Ally','Jane','Belinda'],
    'height':[160,155,163],
    'age': [40,35,42]
    },
    columns = ['name','height','age']
    )

Basic Statistics in Pandas
print(a.describe())




## Module 5 Import/Export Data

import pandas as pd 

Import data

data = pd.read_csv('data/ex_data.csv')
print(data.head())
print(data.ix[1])
print(data.columns)
print(data.index)

sp500 = pd.read_csv('data/sp500.csv',index_col='Symbol',usecols=[0,2,3,7])
print(sp500.head())

Export data
sp500.to_csv('data/test.csv')

from pandas_datareader import data,wb

msft = data.DataReader("MSFT", "yahoo","2017-1-1","2017-1-11")
print(msft.tail())

import quandl
data = quandl.get("FRED/GDP")
print(data.tail())


## Module 6

import datetime

date = datetime.datetime(2017,1,13)
print(date)

import pandas as pd 

date = pd.to_datetime('2017-1-13')
print(date)

date = pd.date_range('2017-1-1',periods=30)
date = pd.date_range('2017-1-1',periods=12,freq='M')
print(date)

import numpy as np
import matplotlib.pyplot as plt

date = pd.date_range('4/29/2015 8:00',periods=600,freq='T')

ts = pd.Series(np.random.randint(0,100,len(date)), index=date)

print(ts.head())
ts.plot()
plt.show()

Re-sampling
ts1 = ts.resample('10min').mean()
ts1.plot()
plt.show()







## Module 7

import sklearn as sk 


from sklearn import datasets

iris = datasets.load_iris()
X,y = iris.data,iris.target
print(iris.target)
print(iris.data)

Supervised Learning: Classfification
#Step 1 Model

from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()

#Step 2 Training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

clf.fit(X_train,y_train)

#Step 3 Testing
y2 = clf.predict(X_test)
print(y2)
print(y_test)

import numpy as np

y3 = clf.predict([[5,3.2,2,4.3]])
print(y3)

Supervised Learning: Regression

import numpy as np 
import matplotlib.pyplot as plt 

X = np.linspace(1,20,100).reshape(-1,1)
y = X + np.random.randn(len(X)).reshape(-1,1)
plt.scatter(X,y)
plt.show()

Step 1: Model

from sklearn import linear_model

lm = linear_model.LinearRegression()

Step 2: Fitting

lm.fit(X,y)

Step 3: Prediction

y2 = lm.predict(X)
plt.plot(X,y2,'-r')
plt.scatter(X,y)
plt.show()

print(lm.predict([[10]]))

Unsupervised Learning: Clustering

iris = datasets.load_iris()
X,y = iris.data,iris.target

Step 1 Mdoel

from sklearn import cluster

clf = cluster.KMeans(n_clusters=3)

Step 2 : Training

clf.fit(X)

Step 3

print(clf.labels_[::10])
print(y[::10])




