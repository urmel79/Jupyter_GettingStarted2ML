---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Introduction

This notebook was basically inspired by:  
- [In Depth: Parameter tuning for SVC](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769)
- [SVM Hyperparameter Tuning using GridSearchCV](https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/):

The goal of this notebook is to show the basic steps in machine learning and the influence of choosing the "right" the kernel of a **support vector classifier (SVC)**. Furthermore, the SVC parameters are described and their effect on the classification result is shown.

Following steps will be shown in next **chapters**:
- [STEP 0: Get the data](#STEP-0:-Get-the-data)
- [STEP 1: Exploring the data](#STEP-1:-Exploring-the-data)
- [STEP 2: Prepare the data](#STEP-2:-Prepare-the-data)
- [STEP 3: Classify by support vector classifier (SVC)](#STEP-3:-Classify-by-support-vector-classifier-(SVC))
- [STEP 4: Evaluate the results (metrics)](#STEP-4:-Evaluate-the-results-(metrics))
- [STEP 5: Vary parameters](#STEP-5:-Vary-parameters)

<!-- #region tags=[] -->
# Load globally used libraries and set plot parameters
<!-- #endregion -->

```python
import time

from IPython.display import HTML
```

<!-- #region tags=[] -->
# STEP 0: Get the data

Since this is intended to be an introduction to the world of machine learning (ML), this step does NOT deal with the design of an application suitable for ML and the acquisition of valid measurement data.

In order to get to know the typical work steps and ML tools, the use of **well-known and well-researched data sets** is clearly **recommended**.

In the further course, the famous [Iris flower data sets](https://en.wikipedia.org/wiki/Iris_flower_data_set) will be used.
It can be downloaded on [Iris Flower Dataset | Kaggle](https://www.kaggle.com/datasets/arshid/iris-flower-dataset). Furthermore, the dataset is included in Python in the machine learning package [Scikit-learn](https://scikit-learn.org), so that users can access it without having to find a special source for it.
<!-- #endregion -->

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
%matplotlib inline

# import some data to play with
irisdata_df = pd.read_csv('./datasets/IRIS_flower_dataset_kaggle.csv')
```

<!-- #region tags=[] toc-hr-collapsed=true -->
# STEP 1: Exploring the data

## Goals of exploration

The objectives of the exploration of the dataset are as follows:

1. Clarify the **origins history**:
    - Where did the data come from? => Contact persons and licensing permissions?
    - Who obtained the data and with which (measurement) methods? => Did systematic errors occur during the acquisition?
    - What were they originally intended for? => Can they be used for my application?

2. Overview of the internal **structure and organisation** of the data:
    - Which columns are there? => With which methods can they be read in (e.g. import of CSV files)?
    - What do they contain for (physical) measured variables? => Which technical or physical correlations exist?
    - Which data formats or types are there? => Do they have to be converted?
    - In which value ranges do the measurement data vary? => Are normalizations necessary?

3. Identify **anomalies** in the data sets:
    - Do the data have **gaps** or **duplicates**? => Does the data set needs to be cleaned?
    - Are there obvious erroneous entries or measurement outliers? => Does (statistical) filtering have to be carried out?

4. Avoidance of **tendencies due to bias**:
    - Are all possible classes included in the dataset and equally distributed? => Does the data set need to be enriched with additional data for balance?

5. Find a first rough **idea of which correlations** could be in the data set
<!-- #endregion -->

## Clarify the **origins history**

> The ***Iris* flower data sets** is a multivariate data set introduced by the British statistician and biologist *Ronald Fisher* in his paper "The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis" (1936). It is sometimes called *Anderson's Iris data set* because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species (source: [Iris flower data set](https://en.wikipedia.org/w/index.php?title=Iris_flower_data_set&oldid=1090001619)).

The dataset is published in Public Domain with a [CC0-License](https://creativecommons.org/share-your-work/public-domain/cc0/).

This dataset became a typical test case for many statistical classification techniques in machine learning such as **support vector machines**.

> [..] measurements of the flowers of fifty plants each of the two species *Iris setosa* and *I. versicolor*, found **growing together in the same colony** and measured by Dr E. Anderson [..] (source: R. A. Fisher (1936). "The use of multiple measurements in taxonomic problems". [Annals of Eugenics](https://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x))

> [..] *Iris virginica*, differs from the two other samples in **not being taken from the same natural colony** [..] (source: ibidem)


## Overview of the internal **structure and organisation** of the data

The data set consists of 50 samples from each of three species of Iris ([*Iris setosa*](https://en.wikipedia.org/wiki/Iris_setosa), [*Iris virginica*](https://en.wikipedia.org/wiki/Iris_virginica) and [*Iris versicolor*](https://en.wikipedia.org/wiki/Iris_versicolor)), so there are 150 total samples. Four features were measured from each sample: the length and the width of the [sepals](https://en.wikipedia.org/wiki/Sepal) and [petals](https://en.wikipedia.org/wiki/Petal), in centimetres.  
Here is a principle illustration of a flower with sepal and petal:

```python tags=[] jupyter={"source_hidden": true}
display(HTML("<figure><img src='./images/Mature_flower_diagram.svg' width='800px'> \
                 <figcaption>Principle illustration of a flower with sepal and petal (source: <a href='https://en.wikipedia.org/wiki/File:Mature_flower_diagram.svg'>Mature_flower_diagram.svg</a>)</figcaption> \
              </figure>"))
```

<!-- #region tags=[] -->
Here are pictures of the three different Iris species (*Iris setosa*, *Iris virginica* and *Iris versicolor*). Given the dimensions of the flower, it will be possible to predict the class of the flower.
<!-- #endregion -->

```python tags=[] jupyter={"source_hidden": true}
display(HTML("<table> \
                <tr> \
                <td><figure><img src='./images/Iris_setosa_640px.jpg' width='320px'> \
                        <figcaption><i>Iris setosa</i> (source: <a href='https://commons.wikimedia.org/wiki/File:Irissetosa1.jpg'>Irissetosa1.jpg</a>)</figcaption> \
                    </figure></td> \
                <td><figure><img src='./images/Iris_versicolor_640px.jpg' width='320px'> \
                        <figcaption><i>Iris versicolor</i> (source: <a href='https://en.wikipedia.org/wiki/File:Iris_versicolor_3.jpg'>Iris versicolor 3.jpg</a>)</figcaption> \
                    </figure></td> \
                <td><figure><img src='./images/Iris_virginica_590px.jpg' width='295px'> \
                        <figcaption><i>Iris virginica</i> (source: <a href='https://en.wikipedia.org/wiki/File:Iris_virginica.jpg'>Iris virginica.jpg</a>)</figcaption> \
                    </figure></td> \
                </tr> \
              </table>"))
```

### Inspect **structure of dataframe**

Print first or last 5 rows of dataframe:

```python
irisdata_df.head()
```

```python
irisdata_df.tail()
```

While printing a dataframe - only an abbreviated view of the dataframe is shown :(  
Default setting in the pandas library makes it to display only 5 lines from head and from tail.

```python tags=[]
irisdata_df
```

To print all rows of a dataframe, the option `display.max_rows` has to set to `None` in pandas:

```python tags=[]
pd.set_option('display.max_rows', None)
irisdata_df
```

### Get data types

```python tags=[]
irisdata_df.info()
```

```python
irisdata_df.describe()
```

### Get data ranges with Boxplots

**Boxplots** can be used to explore the data ranges in the data set. These also provide information about **outliers**.

```python
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

sns.set_style("whitegrid")
#sns.set_style("white")

fn = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
cn = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
box1 = sns.boxplot(x = 'species', y = 'sepal_length', data = irisdata_df, order = cn, ax = axs[0,0])
box2 = sns.boxplot(x = 'species', y = 'sepal_width', data = irisdata_df, order = cn, ax = axs[0,1])
box3 = sns.boxplot(x = 'species', y = 'petal_length', data = irisdata_df, order = cn, ax = axs[1,0])
box4 = sns.boxplot(x = 'species', y = 'petal_width', data = irisdata_df,  order = cn, ax = axs[1,1])

# add some spacing between subplots
fig.tight_layout(pad=2.0)

box1.set_xlabel('species', fontsize = 16)
box1.set_ylabel('sepal length', fontsize = 16)

box2.set_xlabel('species', fontsize = 16)
box2.set_ylabel('sepal width', fontsize = 16)

box3.set_xlabel('species', fontsize = 16)
box3.set_ylabel('petal length', fontsize = 16)

box4.set_xlabel('species', fontsize = 16)
box4.set_ylabel('petal width', fontsize = 16)

plt.show()
```

<!-- #region tags=[] -->
## Identify **anomalies** in the data sets

### Find gaps in dataset

This section was inspired by [Working with Missing Data in Pandas](https://www.geeksforgeeks.org/working-with-missing-data-in-pandas/).

#### Checking for missing values using `isnull()`

In order to check for missing values in Pandas DataFrame, we use the function `isnull()`. This function returns a dataframe of Boolean values which are True for **NaN values**.
<!-- #endregion -->

```python
pd.set_option('display.max_rows', 40)
pd.set_option('display.min_rows', 30)
```

```python tags=[]
irisdata_df.isnull()
```

Show only the gaps:

```python
irisdata_df_gaps = irisdata_df[irisdata_df.isnull().any(axis=1)]
irisdata_df_gaps
```

Fine - this dataset seems to be complete :)

So let's look for something else for exercise: [employes.csv](https://media.geeksforgeeks.org/wp-content/uploads/employees.csv)

```python tags=[]
# import data to dataframe from csv file
employees_df = pd.read_csv("./datasets/employees_edit.csv")

employees_df
```

Show only the gaps from this gappy dataset again:

```python tags=[]
employees_df_gaps = employees_df[employees_df.isnull().any(axis=1)]
employees_df_gaps
```

#### Fill the missing values with `fillna()`

Now we are going to fill all the null (NaN) values in Gender column with *"No Gender"*.

**Attention:** We are doing that directly in this dataframe with `inplace = True` - we don't make a deep copy!

```python tags=[]
# filling a null values using fillna()
employees_df["Gender"].fillna("No Gender", inplace = True)
employees_df
```

#### Dropping missing values using `dropna()`

In order to drop null values from a dataframe, we use `dropna()` function. This function drops rows or columns of datasets with NaN values in different ways.

Default is to drop rows with at least 1 null value (NaN).
Giving the parameter `how = 'all'` the function drops rows with all data missing or contain null values (NaN).

```python tags=[]
# making a new dataframe with dropped NaN values
employees_df_dropped = employees_df.dropna(axis = 0, how ='any')
employees_df_dropped
```

Finally we compare the sizes of dataframes so that we learn how many rows had at least 1 Null value.

```python
print("Old data frame length:", len(employees_df))
print("New data frame length:", len(employees_df_dropped))
print("Number of rows with at least 1 NaN value: ", (len(employees_df)-len(employees_df_dropped)))
```

### Find and remove duplicates in dataset

This section was inspired by:
- [How to Find Duplicates in Pandas DataFrame (With Examples)](https://www.statology.org/pandas-find-duplicates/)
- [How to Drop Duplicate Rows in a Pandas DataFrame](https://www.statology.org/pandas-drop-duplicates/)

#### Checking for duplicate values using `duplicated()`

In order to check for duplicate values in Pandas DataFrame, we use a function `duplicated()`. This function can be used in two ways:
- find duplicate rows across **all columns** with `duplicateRows = df[df.duplicated()]`
- find duplicate rows across **specific columns** `duplicateRows = df[df.duplicated(subset=['col1', 'col2'])]`

Find duplicate rows across **all columns**:

```python
# import (again) data to dataframe from csv file
employees_df = pd.read_csv("./datasets/employees_edit.csv")
```

```python
# find duplicate rows across all columns
duplicateRows = employees_df[employees_df.duplicated()]
duplicateRows
```

```python
# argument keep=’last’ displays the first duplicate rows instead of the last
duplicateRows = employees_df[employees_df.duplicated(keep='last')]
duplicateRows
```

Find duplicate rows across **specific columns**:

```python
# identify duplicate rows across 'First Name' and 'Last Login Time' columns
duplicateRows = employees_df[employees_df.duplicated(subset=['First Name', 'Last Login Time'])]
duplicateRows
```

```python tags=[]
# argument keep=’last’ displays the first duplicate rows instead of the last
duplicateRows = employees_df[employees_df.duplicated(subset=['First Name', 'Last Login Time'], keep='last')]
duplicateRows
```

<!-- #region tags=[] -->
#### Dropping duplicate values using `drop_duplicates()`

In order to drop duplicate values from a dataframe, we use `drop_duplicates()` function.

This function can be used in two ways:
- remove duplicate rows across **all columns** with `df.drop_duplicates()`
- find duplicate rows across **specific columns** `df.drop_duplicates(subset=['col1', 'col2'])`

**Attention:** We are doing that directly in this dataframe with `inplace = True` - we don't make a deep copy!

Remove duplicate rows across **all columns**:
<!-- #endregion -->

```python tags=[]
# remove duplicate rows across all columns
employees_df.drop_duplicates(inplace=True)
employees_df
```

Remove duplicate rows across **specific columns**:

```python tags=[]
# remove duplicate rows across 'First Name' and 'Last Login Time' columns
employees_df.drop_duplicates(subset=['First Name', 'Last Login Time'], keep='last', inplace=True)
employees_df
```

<!-- #region tags=[] -->
## Avoidance of **tendencies due to bias**

The description of the Iris dataset says, that it consists of **50 samples** from **each of three species** of Iris (Iris setosa, Iris virginica and Iris versicolor), so there are **150 total samples**.

But how to prove it?

### Count occurrences of unique values

To prove whether all possible classes included in the dataset and equally distributed, you can use the function `df.value_counts`.

Following parameters can be used for fine tuning:
- `dropna=False` causes that NaN values are included
- `normalize=True`: relative frequencies of the unique values are returned
- `ascending=False`: sort resulting classes descending
<!-- #endregion -->

```python
# import (again) data to dataframe from csv file
employees_df = pd.read_csv("./datasets/employees_edit.csv")
```

```python
# count unique values without missing values in a column, ordered descending and normalized
irisdata_df['species'].value_counts(ascending=False, dropna=False, normalize=True)
```

```python
# count unique values and missing values in a column, ordered descending and not absolute values
employees_df['Team'].value_counts(ascending=False, dropna=False, normalize=False)
```

#### Display Histogram

This section was inspired by: [Pandas Histogram – DataFrame.hist()](https://dataindependent.com/pandas/pandas-histogram-dataframe-hist/).

```python
employees_df.hist(column=['Salary'])
```

```python
employees_df.hist(column='Salary', by='Gender')
```

## First **idea of correlations** in data set

To get a rough idea of the **dependencies** and **correlations** in the data set, it can be helpful to visualize the whole dataset in a **correlation heatmap**. They show in a glance which variables are correlated, to what degree and in which direction.

Later, 2 particularly well correlated variables are selected from the data set and plotted in a **scatterplot**.

<!-- #region -->
### Visualise data with **correlation heatmap**

This section was inspired by [How to Create a Seaborn Correlation Heatmap in Python?](https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e).


> **Correlation matrices** are an **essential tool of exploratory data analysis**. Correlation heatmaps contain the same information in a visually appealing way. What more: they show in a glance which variables are correlated, to what degree, in which direction, and alerts us to potential multicollinearity problems (source: ibidem).

#### Simple correlation matrix

Because **string values can never be correlated**, the class names (species) have to be converted first:
<!-- #endregion -->

```python
# encoding the class column
irisdata_df_enc = irisdata_df.replace({"species":  {"Iris-setosa":1,"Iris-versicolor":2, "Iris-virginica":3}})
irisdata_df_enc
```

```python
irisdata_df_enc.corr()
```

#### Correlation heatmap

Choose the color sets from [color map](https://pod.hatenablog.com/entry/2018/09/20/212527).

```python
# increase the size of the heatmap
plt.figure(figsize=(16, 6))

# store heatmap object in a variable to easily access it 
# when you want to include more features (such as title)
# set the range of values to be displayed on the colormap from -1 to 1,
# and set 'annotation=True' to display the correlation values on the heatmap
heatmap = sns.heatmap(irisdata_df_enc.corr(), vmin=-1, vmax=1, annot=True, cmap='PRGn_r')

# give a title to the heatmap
# 'pad=12' defines the distance of the title from the top of the heatmap
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
```

#### Triangle correlation heatmap

When looking at the correlation heatmaps above, you would not lose any information by **cutting** away half of it **along the diagonal** line marked by 1-s.

The **numpy** function `np.triu()` can be used to isolate the upper triangle of a matrix while turning all the values in the lower triangle into 0.

```python
import numpy as np

np.triu(np.ones_like(irisdata_df_enc.corr()))
```

Use this mask to cut the heatmap along the diagonal:

```python
plt.figure(figsize=(16, 6))

# define the mask to set the values in the upper triangle to 'True'
mask = np.triu(np.ones_like(irisdata_df_enc.corr(), dtype=bool))

heatmap = sns.heatmap(irisdata_df_enc.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='PRGn_r')

heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
```

As a result from the **heatmaps** we can see, that the shape of the **petals** are the **most correlationed columns** (0.96) with the **type of flowers** (species classes).

Somewhat lower correlates **sepal length** with **petal length** (0.87).

<!-- #region tags=[] -->
### Visualise data with **scatter plot**

In the following, [Seaborn](https://seaborn.pydata.org/) is applied which is a library for making statistical graphics in Python. It is built on top of matplotlib and closely integrated with pandas data structures. 

To investigate whether there are dependencies (e.g. correlations) in `irisdata_df` between individual variables in the data set, it is advisable to plot them in a **scatter plot**.
<!-- #endregion -->

```python
# There are five preset seaborn themes: darkgrid, whitegrid, dark, white, and ticks.
sns.set_style("whitegrid")
# set scale of fonts
sns.set_context("notebook", font_scale=1.3, rc={"lines.linewidth": 2.5})

# 'sepal_length', 'petal_length' are iris feature data
# 'height' used to define height of graph
# 'hue' stores the class/label of iris dataset
sns.FacetGrid(irisdata_df, hue ="species",
              height = 7).map(plt.scatter,
                              'petal_width',
                              'petal_length').add_legend()
```

### Visualise data with **pairs plot**

For systematic investigation of dependencies, all variables (each against each) are plotted in separate scatter plots.

With this so called **[pairs plot](https://vita.had.co.nz/papers/gpp.pdf)** it is possible to see both **relationships** between two variables and **distribution** of single variables.

This function will create a grid of Axes such that **each numeric variable** in `irisdata_df` will by shared in the y-axis across a single row and in the x-axis across a single column.

```python
sns.set_style("white")
g = sns.pairplot(irisdata_df, diag_kind="kde", hue='species', palette='Dark2')
g.map_lower(sns.kdeplot, levels=4, color=".2")
```

# STEP 2: Prepare the data

Through the intensive exploration of the data in Step 1 ([STEP 1: Exploring the data](#STEP-1:-Exploring-the-data)), we know that special **preparation** of the data is **not necessary**. The values are **complete** and **without gaps** and there are **no duplicates**. The values are in similar ranges, which **does not require** **normalization** of the data.

Furthermore, we know that the **classes** are very **evenly distributed** and thus bias tendencies should be avoided.

<!-- #region tags=[] toc-hr-collapsed=true -->
# STEP 3: Classify by support vector classifier (SVC)

## Operating principal

> Support Vectors Classifier tries to **find the best hyperplane to separate** the different classes by maximizing the distance between sample points and the hyperplane (source: [In Depth: Parameter tuning for SVC](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769)).

Following graphic shows the operating principal of SVC: the hyperplane *H1* does not separate the classes. *H2* does, but only with a small margin. *H3* separates them with the maximal margin (source: [Support-vector machine](https://en.wikipedia.org/wiki/Support-vector_machine)).
<!-- #endregion -->

```python jupyter={"source_hidden": true} tags=[]
display(HTML("<figure><img src='./images/SVM_separating_hyperplanes.svg' width='400px'> \
                 <figcaption>SVC seperate the data in classes by finding the best hyperplane (source: <a href='https://en.wikipedia.org/wiki/File:Svm_separating_hyperplanes_(SVG).svg'>Svm separating hyperplanes (SVG).svg</a>)</figcaption> \
              </figure>"))
```

## Split the dataset

In the next very important step, the dataset is split into **2 subsets**: a **training dataset** and a **test dataset**. As the names suggest, the training dataset is used to train the ML algorithm. The test data set is then used to check the quality of the trained ML algorithm (here the **recognition rate**). For this purpose, the **class labels** are **removed** from the training data set - after all, these are to be predicted.

Typically, the **test dataset** set should contain **20%** of the entire dataset.

```python
from sklearn.model_selection import train_test_split

X = irisdata_df.drop('species', axis=1)
y = irisdata_df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
```

## Create the SVM model

In this step we create the SVC model and fit it to our training data.

```python
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)

# fit the model for the data
classifier.fit(X_train, y_train)
```

## Make predictions

```python
y_pred = classifier.predict(X_test)
#X_test
```

# STEP 4: Evaluate the results (metrics)

And finally for checking the accuracy of the model, the **confusion matrix** is used for the **cross validation**.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
```

```python
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
```

# STEP 5: Vary parameters

This section was inspired by [In Depth: Parameter tuning for SVC](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769)

## Plotting function

This function helps to visualize the modifications by varying the individual parameters.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
%matplotlib inline

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]

# we only take the first two features
# we could avoid this ugly slicing by using a two-dim dataset
y = iris.target
```

```python tags=[]
y = irisdata_df_enc['species']
y
```

```python tags=[]
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

print(xx, yy)
```

```python
x_min, x_max = X.min()['sepal_length'] - 1, X.max()['sepal_length'] + 1
y_min, y_max = X.min()['sepal_width'] - 1, X.min()['sepal_width'] + 1

h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

print(xx, yy)
```

```python
svc = svm.SVC(kernel=kernel).fit(X[['sepal_length', 'sepal_width']], y)
```

```python
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
```

```python
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
```

```python
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
```

```python
def plotSVC(title, svc):
    # create a mesh to plot in
    #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    x_min, x_max = X.min()['sepal_length'] - 1, X.max()['sepal_length'] + 1
    y_min, y_max = X.min()['sepal_width'] - 1, X.min()['sepal_width'] + 1
    
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z = svc.predict(X_test)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title(title)
    plt.show()
```

## Vary `kernel`

```python
kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    #svc = svm.SVC(kernel=kernel).fit(X, y)
    svc = svm.SVC(kernel=kernel).fit(X[['sepal_length', 'sepal_width']], y)
    
    plotSVC('kernel=' + str(kernel), svc)
```

## Vary `gamma`

```python
gammas = [0.1, 1, 10, 100]

for gamma in gammas:
    svc = svm.SVC(kernel='rbf', gamma=gamma).fit(X, y)
    plotSVC('gamma=' + str(gamma))
```

## Vary `C`

```python
cs = [0.1, 1, 10, 100, 1000]

for c in cs:
    svc = svm.SVC(kernel='rbf', C=c).fit(X, y)
    plotSVC('C=' + str(c))
```

```python

```
