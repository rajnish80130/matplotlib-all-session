import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Colored Scatterplots
'''iris = pd.read_csv(r'matplotlib\iris.csv')
print(iris.sample(5))

iris['Species'] = iris['Species'].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
print(iris.sample(5))

# cmap and alpha
plt.scatter(iris['SepalLengthCm'],iris['PetalLengthCm'],c=iris['Species'],cmap='jet',alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.colorbar()
plt.show()

# Plot size
plt.figure(figsize=(15,7))

plt.scatter(iris['SepalLengthCm'],iris['PetalLengthCm'],c=iris['Species'],cmap='jet',alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.colorbar()
plt.show()

# Annotations
batters = pd.read_csv(r'matplotlib\batter.csv')
print(batters.shape)

sample_df = batters.head(100).sample(25,random_state=5)
print(sample_df)

plt.figure(figsize=(18,10))
plt.scatter(sample_df['avg'],sample_df['strike_rate'],s=sample_df['runs'])

for i in range(sample_df.shape[0]):
  plt.text(sample_df['avg'].values[i],sample_df['strike_rate'].values[i],sample_df['batter'].values[i])

plt.show()
###########
x = [1,2,3,4]
y = [5,6,7,8]

plt.scatter(x,y)
plt.text(1,5,'Point 1')
plt.text(2,6,'Point 2')
plt.text(3,7,'Point 3')
plt.text(4,8,'Point 4',fontdict={'size':12,'color':'brown'})
plt.show()

# Horizontal and Vertical lines
plt.figure(figsize=(18,10))
plt.scatter(sample_df['avg'],sample_df['strike_rate'],s=sample_df['runs'])

plt.axhline(130,color='red')
plt.axhline(140,color='green')
plt.axvline(30,color='red')

for i in range(sample_df.shape[0]):
  plt.text(sample_df['avg'].values[i],sample_df['strike_rate'].values[i],sample_df['batter'].values[i])

plt.show()

# Subplots
# A diff way to plot graphs

print(batters.head())

plt.figure(figsize=(15,6))
plt.scatter(batters['avg'],batters['strike_rate'])
plt.title('Something')
plt.xlabel('Avg')
plt.ylabel('Strike Rate')

plt.show()
##############
fig,ax = plt.subplots(figsize=(15,6))

ax.scatter(batters['avg'],batters['strike_rate'],color='red',marker='+')
ax.set_title('Something')
ax.set_xlabel('Avg')
ax.set_ylabel('Strike Rate')

plt.show()

# batter dataset
fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(10,6))

ax[0].scatter(batters['avg'],batters['strike_rate'],color='red')
ax[1].scatter(batters['avg'],batters['runs'])

ax[0].set_title('Avg Vs Strike Rate')
ax[0].set_ylabel('Strike Rate')


ax[1].set_title('Avg Vs Runs')
ax[1].set_ylabel('Runs')
ax[1].set_xlabel('Avg')

plt.show()
###############
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,10))

ax[0,0].scatter(batters['avg'],batters['strike_rate'])
ax[0,1].scatter(batters['avg'],batters['runs'])
ax[1,0].hist(batters['avg'])
ax[1,1].hist(batters['runs'])

plt.show()
###############
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.scatter(batters['avg'],batters['strike_rate'],color='red')

ax2 = fig.add_subplot(2,2,2)
ax2.hist(batters['runs'])

ax3 = fig.add_subplot(2,2,3)
ax3.hist(batters['avg'])

plt.show()

#########
fig, ax = plt.subplots(nrows=2,ncols=2,sharex=True,figsize=(10,10))

ax[1,1]
plt.show()

# 3D Scatter Plots
batters = pd.read_csv(r'matplotlib\batter.csv')
print(batters)

fig = plt.figure()

ax = plt.subplot(projection='3d')

ax.scatter3D(batters['runs'],batters['avg'],batters['strike_rate'],marker='+')
ax.set_title('IPL batsman analysis')

ax.set_xlabel('Runs')
ax.set_ylabel('Avg')
ax.set_zlabel('SR')
plt.show()

# 3D Line Plot
x = [0,1,5,25]
y = [0,10,13,0]
z = [0,13,20,9]

fig = plt.figure()

ax = plt.subplot(projection='3d')

ax.scatter3D(x,y,z,s=[100,100,100,100])
ax.plot3D(x,y,z,color='red')
plt.show()

# 3D Surface Plots

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)

xx, yy = np.meshgrid(x,y)

print(xx,'##################',yy)

z = xx**2 + yy**2
print(z.shape)

fig = plt.figure(figsize=(12,8))

ax = plt.subplot(projection='3d')

p = ax.plot_surface(xx,yy,z,cmap='viridis')
fig.colorbar(p)
plt.show()

####################

z = np.sin(xx) + np.cos(yy)

fig = plt.figure(figsize=(12,8))

ax = plt.subplot(projection='3d')

p = ax.plot_surface(xx,yy,z,cmap='viridis')
fig.colorbar(p)
plt.show()
##############
z = np.sin(xx) + np.log(xx)

fig = plt.figure(figsize=(12,8))

ax = plt.subplot(projection='3d')

p = ax.plot_surface(xx,yy,z,cmap='viridis')
fig.colorbar(p)
plt.show()

# Contour Plots
fig = plt.figure(figsize=(12,8))

ax = plt.subplot()

p = ax.contour(xx,yy,z,cmap='viridis')
fig.colorbar(p)
plt.show()

####################
fig = plt.figure(figsize=(12,8))

ax = plt.subplot()

p = ax.contourf(xx,yy,z,cmap='viridis')
fig.colorbar(p)
plt.show()

#######################
z = np.sin(xx) + np.cos(yy)

fig = plt.figure(figsize=(12,8))

ax = plt.subplot()

p = ax.contourf(xx,yy,z,cmap='viridis')
fig.colorbar(p)
plt.show()

# Heatmap

delivery = pd.read_csv(r'matplotlib\IPL_Ball_by_Ball_2008_2022.csv')
print(delivery.head())

temp_df = delivery[(delivery['ballnumber'].isin([1,2,3,4,5,6])) & (delivery['batsman_run']==6)]
print(temp_df)

grid = temp_df.pivot_table(index='overs',columns='ballnumber',values='batsman_run',aggfunc='count')
print(grid)

plt.figure(figsize=(20,10))
plt.imshow(grid)
plt.yticks(delivery['overs'].unique(), list(range(1,21)))
plt.xticks(np.arange(0,6), list(range(1,7)))
plt.colorbar()
plt.show()

# Pandas Plot()
# on a series

s = pd.Series([1,2,3,4,5,6,7])
s.plot(kind='pie')
plt.show()

# can be used on a dataframe as well
import seaborn as sns
tips = sns.load_dataset('tips')
print(tips)

tips['size'] = tips['size'] * 100

print(tips.head())

# Scatter plot -> labels -> markers -> figsize -> color -> cmap
tips.plot(kind='scatter',x='total_bill',y='tip',title='Cost Analysis',marker='+',figsize=(10,6),s='size',c='sex',cmap='viridis')
plt.show()

# 2d plot
# dataset = 'https://raw.githubusercontent.com/m-mehdi/pandas_tutorials/main/weekly_stocks.csv'

stocks = pd.read_csv('https://raw.githubusercontent.com/m-mehdi/pandas_tutorials/main/weekly_stocks.csv')
print(stocks.head())

# line plot
stocks['MSFT'].plot(kind='line')
plt.show()

###########
stocks.plot(kind='line',x='Date')
plt.show()

############
stocks[['Date','AAPL','FB']].plot(kind='line',x='Date')
plt.show()

# bar chart -> single -> horizontal -> multiple
# using tips
temp = pd.read_csv(r'matplotlib\batsman_season_record.csv')
print(temp.head())

import seaborn as sns
tips = sns.load_dataset('tips')
print(tips)

tips.groupby('sex')['total_bill'].mean().plot(kind='bar')
plt.show()

################
temp['2015'].plot(kind='bar')
plt.show()

###############
temp.plot(kind='bar')
plt.show()

##################
# stacked bar chart
temp.plot(kind='bar',stacked=True)
plt.show()

# histogram
# using stocks
stocks = pd.read_csv('https://raw.githubusercontent.com/m-mehdi/pandas_tutorials/main/weekly_stocks.csv')

stocks[['MSFT','FB']].plot(kind='hist',bins=40)
plt.show()

# pie -> single and multiple
df = pd.DataFrame(
    {
        'batsman':['Dhawan','Rohit','Kohli','SKY','Pandya','Pant'],
        'match1':[120,90,35,45,12,10],
        'match2':[0,1,123,130,34,45],
        'match3':[50,24,145,45,10,90]
    }
)

print(df.head())

df['match1'].plot(kind='pie',labels=df['batsman'].values,autopct='%0.1f%%')
plt.show()

# multiple pie charts

df[['match1','match2','match3']].plot(kind='pie',subplots=True,figsize=(15,8))
plt.show()

# multiple separate graphs together
# using stocks

stocks.plot(kind='line',subplots=True)
plt.show()

# on multiindex dataframes
# using tips

import seaborn as sns
tips = sns.load_dataset('tips')
print(tips)
tips.pivot_table(index=['day','time'],columns=['sex','smoker'],values='total_bill',aggfunc='mean').plot(kind='pie',subplots=True,figsize=(20,10))
plt.show()'''