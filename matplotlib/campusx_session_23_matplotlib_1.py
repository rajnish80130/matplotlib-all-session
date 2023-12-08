# Types of Data
# 1)Numerical Data
# 2)Categorical Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')

# 2D Line plot
# Bivariate Analysis
# categorical -> numerical and numerical -> numerical
# Use case - Time series data

# plotting a simple function
price = [48000, 54000, 57000, 49000, 47000, 45000]
year = [2015, 2016, 2017, 2018, 2019, 2020]

plt.plot(year, price)
plt.show()

# from a pandas dataframe
batsman = pd.read_csv(r'matplotlib\sharma-kohli.csv')
print(batsman)

plt.plot(batsman['index'], batsman['V Kohli'])
plt.show()

# plotting multiple plots
plt.plot(batsman['index'], batsman['V Kohli'])
plt.plot(batsman['index'], batsman['RG Sharma'])

plt.show()

# labels title

plt.plot(batsman['index'], batsman['V Kohli'])
plt.plot(batsman['index'], batsman['RG Sharma'])

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')

plt.show()

# colors(hex) and line(width and style) and marker(size)

batsman = pd.read_csv(r'matplotlib\sharma-kohli.csv')
print(batsman)

plt.plot(batsman['index'], batsman['V Kohli'],
         color='#D9F10F', linestyle='solid', linewidth=3)
plt.plot(batsman['index'], batsman['RG Sharma'],
         color='#FC00D6', linestyle='dashdot', linewidth=2)

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')
plt.show()

#############

plt.plot(batsman['index'], batsman['V Kohli'], color='#D9F10F',
         linestyle='solid', linewidth=3, marker='D', markersize=10)
plt.plot(batsman['index'], batsman['RG Sharma'], color='#FC00D6',
         linestyle='dashdot', linewidth=2, marker='o', markersize=8)

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')
plt.show()

# legend -> location
plt.plot(batsman['index'], batsman['V Kohli'], color='#D9F10F',
         linestyle='solid', linewidth=3, marker='D', markersize=10, label='Virat')
plt.plot(batsman['index'], batsman['RG Sharma'], color='#FC00D6',
         linestyle='dashdot', linewidth=2, marker='o', label='Rohit')

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')

plt.legend()  # here you can also specific any type like (loc='upper right')
plt.show()

# limiting axes
price = [48000, 54000, 57000, 49000, 47000, 45000, 4500000]
year = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

plt.plot(year, price)
plt.ylim(0, 75000)
plt.xlim(2017, 2019)
plt.show()

# grid
price = [48000, 54000, 57000, 49000, 47000, 45000, 4500000]
year = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

plt.plot(year, price)
plt.ylim(0, 75000)
plt.xlim(2017, 2019)

plt.grid()
plt.show()

# show
price = [48000, 54000, 57000, 49000, 47000, 45000, 4500000]
year = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

plt.plot(year, price)
plt.ylim(0, 75000)
plt.xlim(2017, 2019)

plt.grid()
plt.show()

# Scatter Plots
# Bivariate Analysis
# numerical vs numerical
# Use case - Finding correlation

# plt.scatter simple function
x = np.linspace(-10, 10, 50)

y = 10*x + 3 + np.random.randint(0, 300, 50)
print(y)

plt.scatter(x, y)
plt.show()

# plt.scatter on pandas data
df = pd.read_csv(r'matplotlib\batter.csv')
df = df.head(50)
print(df)

# marker
plt.scatter(df['avg'], df['strike_rate'], color='red', marker='+')
plt.title('Avg and SR analysis of Top 50 Batsman')
plt.xlabel('Average')
plt.ylabel('SR')
plt.show()

# size
tips = sns.load_dataset('tips')
print(tips)
# slower
plt.scatter(tips['total_bill'], tips['tip'], s=tips['size']*20)
plt.show()

# plt.plot vs plt.scatter
# scatterplot using plt.plot
# faster but you donot access all function
plt.plot(tips['total_bill'], tips['tip'], 'o')
plt.show()

# Bar chart
# Bivariate Analysis
# Numerical vs Categorical
# Use case - Aggregate analysis of groups

# simple bar chart
children = [10, 20, 40, 10, 30]
colors = ['red', 'blue', 'green', 'yellow', 'pink']

plt.bar(colors, children, color='black')
plt.show()

# horizontal bar chart
plt.barh(colors, children, color='black')
plt.show()

# color and label
df = pd.read_csv(r'matplotlib\batsman_season_record.csv')
print(df)

# Multiple Bar charts
# xticks
plt.bar(np.arange(df.shape[0]) - 0.2, df['2015'], width=0.2, color='yellow')
plt.bar(np.arange(df.shape[0]), df['2016'], width=0.2, color='red')
plt.bar(np.arange(df.shape[0]) + 0.2, df['2017'], width=0.2, color='blue')

plt.xticks(np.arange(df.shape[0]), df['batsman'])

plt.show()

# a problem
children = [10, 20, 40, 10, 30]
colors = ['red red red red red red', 'blue blue blue blue',
          'green green green green green', 'yellow yellow yellow yellow ', 'pink pinkpinkpink']

plt.bar(colors, children, color='black')
plt.xticks(rotation='vertical')
plt.show()

# Stacked Bar chart
plt.bar(df['batsman'], df['2017'], label='2017')
plt.bar(df['batsman'], df['2016'], bottom=df['2017'], label='2016')
plt.bar(df['batsman'], df['2015'], bottom=(
    df['2016'] + df['2017']), label='2015')

plt.legend()
plt.show()

# Histogram
# Univariate Analysis
# Numerical col
# Use case - Frequency Count

# simple data
data = [32, 45, 56, 10, 15, 27, 61]

plt.hist(data, bins=[10, 25, 40, 55, 70])
plt.show()

# on some data
df = pd.read_csv(r'matplotlib\vk.csv')
print(df)

# handling bins
plt.hist(df['batsman_runs'], bins=[0, 10, 20, 30,
         40, 50, 60, 70, 80, 90, 100, 110, 120])
plt.show()

# logarithmic scale
arr = np.load(r'matplotlib\big-array.npy')
plt.hist(arr, bins=[10, 20, 30, 40, 50, 60, 70], log=True)
plt.show()

# Pie Chart
# Univariate/Bivariate Analysis
# Categorical vs numerical
# Use case - To find contibution on a standard scale

# simple data
data = [23, 45, 100, 20, 49]
subjects = ['eng', 'science', 'maths', 'sst', 'hindi']
plt.pie(data, labels=subjects)

plt.show()

# dataset
df = pd.read_csv(r'matplotlib\gayle-175.csv')
print(df)

plt.pie(df['batsman_runs'], labels=df['batsman'], autopct='%0.1f%%')
plt.show()

# percentage and colors
plt.pie(df['batsman_runs'], labels=df['batsman'], autopct='%0.1f%%',
        colors=['blue', 'green', 'yellow', 'pink', 'cyan', 'brown'])
plt.show()

# explode and shadow
plt.pie(df['batsman_runs'], labels=df['batsman'], autopct='%0.1f%%',
        explode=[0.3, 0, 0, 0, 0, 0.1], shadow=True)
plt.show()

# Changing styles
print(plt.style.available)

plt.style.use('dark_background')

arr = np.load(r'matplotlib\big-array.npy')
plt.hist(arr, bins=[10, 20, 30, 40, 50, 60, 70], log=True)
plt.show()

# Save figure
arr = np.load(r'matplotlib\big-array.npy')
plt.hist(arr, bins=[10, 20, 30, 40, 50, 60, 70], log=True)

plt.savefig('sample.png')
