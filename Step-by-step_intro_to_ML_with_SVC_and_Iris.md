---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Introduction

<!-- #region tags=[] -->
## English introduction

In the **digitised work environment**, there is an increasing demand for **Work equipment** to be able to adapt independently and in a task-related manner to changing work situations. This **situational adaptivity** can often only be realised through the use of **Artificial Intelligence (AI)** or **Machine Learning (ML)**, depending on the degree of flexibility.
Examples of such AI applications in the world of work can range from comparatively simple **voice assistance systems** (similar, for example, to Siri or Alexa from the private sphere) to partially or even **fully autonomous systems**. Such fully autonomous systems are, for example, autonomously driving logistics vehicles in larger industrial plants (so-called **driverless transport systems**).

In addition to the many very interesting advantages in terms of economic efficiency, workload reduction, etc., such fully autonomous systems are characterised by a very high level of technical complexity. This concerns both their **operating functions** (e.g. autonomous navigation through complex industrial environments with shared use of the roadways by other human-controlled vehicles) and their **safety functions** (e.g. evaluation of complex, interconnected, mostly imaging safety sensors for monitoring the driving space).

Very high demands are placed on such autonomous systems and the AI algorithms used for them with regard to **functional safety**. However, when assessing their safety, one quickly comes up against clear limits with regard to the **transparency** and **explainability** of the decisions made by AI as well as limits to the **recognition rates** and thus their **reliability**. In particular, the detection rates achievable by AI even under the most convenient conditions very often do not meet the requirements for realising higher safety levels (e.g. Performance Level d (PLd) according to ISO 13849).

An appropriate assessment or even **testing** with regard to the required functional safety according to uniform and ideally standardised criteria has many implications for the future orientation of technical **occupational safety and health (OSH)** in Germany and in Europe. In addition to the currently still very difficult algorithmic evaluability, a significant aspect is that the previous clear separation between **placing on the market law** (see e.g. Machinery Directive) and **occupational health and safety law** (see European Occupational Health and Safety Framework Directive and German Ordinance on Occupational Safety and Health) can no longer be continued in this way. The reason for this is that the **safety-relevant properties** of the autonomous systems will change due to new or **adapted behaviours** learned during operation.

For these reasons, those involved in technical occupational safety and health who will be involved in the testing of work equipment in the future should deal with AI and ML algorithms in depth as early as possible. This is the only way to ensure that the rapid development of adaptive systems capable of learning can be accompanied by OSH and its testing institutes in a constructive, critical and technically appropriate manner. If this is not done, the OSH system will be ruthlessly circumvented or undermined by the economic interests of globally operating software giants. This would have the consequence that serious or fatal occupational accidents are likely to occur due to inadequately designed AI-based work systems.

Anyone seeking a serious technical entrance into the world of **Artificial Intelligence (AI)** or **Machine Learning (ML)** will not be able to avoid dealing with the basic ML algorithms, corresponding software tools, libraries and programming systems.

However, someone who opens the door for the first time to this equally very exciting as well as arbitrarily complex and, at first glance, confusing world will very quickly be overwhelmed. Here, it is a good idea to consult introductory and systematic tutorials.

The aim of this Getting Started tutorial is to systematically demonstrate the typical ML working process step-by-step based on the example of the very powerful and performant **Support Vector Classifier (SVC)**.

This tutorial will be presented as part of a workshop at the DGUV symposium **Artificial Intelligence**, probably in November 2022 in Dresden. The workshop addresses interested ML novices in the technical occupational safety and health of the social accident insurance institutions.

For the target audience in the workshop, the SVC algorithm was intentionally chosen to show that there are many other very powerful and performant ML algorithms apart from the **deep neural networks** that are very present in the media. On the other hand, a necessary and comprehensible introduction to neural networks and the the technical background to perceptrons, activation functions etc. for newcomers would not be possible within the time frame given for the workshop.

Furthermore, this tutorial does *not* address the generation or acquisition of ML-ready datasets. Reason for this is that a newcomer to ML will (or should) first try to familiarize himself with ML algorithms, tools, libraries and programming systems. Only then it makes sense to explore one's own environment with respect to ML-suitable applications and to acquire suitable datasets from them.

Therefore, this tutorial demonstrates the usage of selected ML tools in the form of Python libraries as well as the systematic approach to the widely known and very beginner-friendly **Iris dataset**. According to the literature, the Support Vector Classifier is particularly well suited for the classification of the iris dataset in terms of recognition rate and performance. Alternatively, decision tree-based ML algorithms such as the **Random Forests Classifier** could be used.

After the classification of the iris dataset by the SVC initially with standard parameters, the selection of the "correct" SVC kernel with its setting parameters is furthermore described and the effect on the classification result is shown.
<!-- #endregion -->

<!-- #region tags=[] -->
## German introduction

Von den **Arbeitsmitteln** in der **digitalisierten Arbeitswelt** wird immer stärker gefordert, dass sie sich selbstständig und aufgabenbezogen an sich ändernde Arbeitssituationen anpassen können. Diese **situative Adaptivität** kann je nach Stärke des Flexibilisierungsgrades oft nur durch Anwendung von **Artificial Intelligence (AI)** oder **Machine Learning (ML)** realisiert werden.

Als Beispiele für solche KI-Anwendungen in der Arbeitswelt können vergleichsweise einfache **Sprachassistenzsysteme** (ähnlich z. B. Siri oder Alexa aus dem privaten Umfeld) bis hin zu teil- oder gar **vollautonomen Systemen** genannt werden. Solche vollautonomen Systemen sind beispielsweise autonom fahrende Logistikfahrzeuge in größeren Industrieanlagen (sog. **fahrerlosen Transportsystemen**).

Neben den vielen sehr interessanten Vorteilen bzgl. Wirtschaftlichkeit, Arbeitserleichterung usw. kennzeichnet solche vollautonomen Systeme eine sehr hohe technische Komplexität. Diese betrifft sowohl ihre **Betriebsfunktionen** (z. B. autonome Navigation durch komplexe industrielle Umgebungen bei gemeinsamer Nutzung der Fahrwege durch andere menschlich gesteuerte Fahrzeuge) als auch seiner **Sicherheitsfunktionen** (z. B. Auswertung komplexer, miteinander verknüpfter, meist bildgebender Sicherheitssensorik zur Überwachung des Fahrraums).

An solche autonomen Systeme und die hierfür eingesetzten KI-Algorithmen werden sehr hohe Anforderungen hinsichtlich der **funktionalen Sicherheit** gestellt. Jedoch stößt man bei ihrer sicherheitstechnischen Bewertung heute noch sehr schnell an deutliche Grenzen hinsichtlich der **Transparenz** und **Erklärbarkeit** der durch KI getroffenen Entscheidungen sowie Grenzen der **Erkennnungsraten** und damit ihrer **Zuverlässigkeit**. Insbesondere erfüllen die durch KI selbst unter günstigsten Bedingungen erreichbaren Erkennnungsraten sehr oft nicht die Anforderderungen, um höhere Safety-Level (z. B. Performance Level d (PLd) nach ISO 13849) zu realisieren.

Eine hinsichtlich der geforderten funktionalen Sicherheit angemessene Bewertung oder gar **Prüfung** nach einheitlichen und idealerweise genormten Maßstäben hat viele Implikationen auf die zukünftige Ausrichtung des **technischen Arbeitsschutzes** in Deutschland und in Europa. Neben der derzeit noch sehr schwierigen algorithmischen Bewertbarkeit ist ein wesentlicher Aspekt, dass die bisherige klare Trennung zwischen **Inverkehrbringensrecht** (siehe z. B. Maschinenrichtlinie) und **betrieblichem Arbeitsschutzrecht** (siehe Arbeitschutzrahmenrichtlinie und Betriebssicherheitsverordnung) so nicht mehr aufrechterhalten werden kann. Grund hierfür ist, dass sich die **sicherheitsrelevanten Eigenschaften** der autonomen Systeme durch während des Betriebs erlernte, neue oder **angepasste Verhaltensweisen** verändern werden.

Aus diesen Gründen sollten sich insbesondere die zukünftig mit der Prüfung befassten Akteure des technischen Arbeitsschutzes möglichst frühzeitig mit den KI- bzw. ML-Algorithmen vertieft auseinandersetzen. Nur dadurch lässt sich erreichen, dass die stürmische Entwicklung lernfähiger, adaptiver Systeme durch den Arbeitsschutz und deren Prüfinstitute konstruktiv, kritisch und fachlich angemessen begleitet werden kann. Wird dies versäumt, wird das Arbeitsschutzsystem durch die wirtschaftlichen Interessen global agierender Softwaregiganten skrupellos umgangen oder ausgehebelt werden. Dies hätte die Folge, dass schwere oder tödliche Arbeitsunfälle auf Grund unzulänglich gestalteter KI-basierter Arbeitssysteme wahrscheinlich werden.

Wer einen ernsthaften fachlichen Einstieg in die Welt von **Künstlicher Intelligenz (KI)** bzw. **Machine Learning (ML)** sucht, wird nicht umhin kommen, sich mit den grundlegenden ML-Algorithmen, entsprechenden Software-Werkzeugen, Bibliotheken und Programmiersystemen auseinander zu setzen.

Wer jedoch zum ersten Mal die Tür zu dieser ebenso spannenden wie beliebig komplexen und auf den ersten Blick verwirrenden Welt öffnet, wird sehr schnell überfordert sein. Hier empfiehlt es sich, einführende und systematische Anleitungen zu Rate zu ziehen.

Ziel dieses Getting-Started-Tutorials ist es, den typischen ML-Arbeitsablauf systematisch und Schritt-für-Schritt am Beispiel des sehr leistungsfähigen **Support Vector Classifier (SVC)** zu demonstrieren.

Dieses Tutorial wird im Rahmen eines Workshops auf der DGUV-Fachtagung **Künstliche Intelligenz** voraussichtlich im November 2022 in Dresden vorgestellt. Der Workshop richtet sich an interessierte ML-Neulinge im technischen Arbeitsschutz der gesetzlichen Unfallversicherungsträger.

Für die Zielgruppe des Workshops wurde der SVC-Algorithmus bewusst gewählt, um zu zeigen, dass es neben den **tiefen neuronalen Netzen**, die in den Medien sehr präsent sind, noch viele andere sehr leistungsfähige ML-Algorithmen gibt. Andererseits wäre eine notwendige und verständliche Einführung in neuronale Netze und die technischen Hintergründe zu Perzeptronen, Aktivierungsfunktionen etc. für Neulinge in dem für den Workshop vorgegebenen Zeitrahmen nicht möglich gewesen.

Außerdem befasst sich dieses Tutorial *nicht* mit der Erzeugung oder Akquisition von ML-tauglichen Datensätzen. Der Grund dafür ist, dass ein ML-Neuling zunächst versuchen wird (oder sollte), sich mit den ML-Algorithmen, Werkzeugen, Bibliotheken und Programmiersystemen vertraut zu machen. Erst dann ist es sinnvoll, die eigene Umgebung auf ML-taugliche Anwendungen hin zu untersuchen und daraus geeignete Datensätze zu gewinnen.

Daher demonstriert dieses Tutorial die Verwendung ausgewählter ML-Tools in Form von Python-Bibliotheken sowie die systematische Herangehensweise an den weithin bekannten und sehr einsteigerfreundlichen **Iris-Datensatz**. Laut Fachliteratur ist für die Klassifikation des Iris-Datensatzes der Support Vector Classifier hinsichtlich Erkennungsrate als auch Performanz besonders gut geeignet. Alternativ könnten auch entscheidungsbaum-basierte ML-Algorithmen wie z. B. der **Random-forests-Klassifikator** eingesetzt werden.

Nach der Klassifikation des Iris-Datensatzes durch den SVC zunächst mit Standard-Parametern wird darüber hinaus die Auswahl des "richtigen" SVC-Kernels mit seinen Einstellparametern beschrieben und die Auswirkung auf das Klassifikationsergebnis wird gezeigt.
<!-- #endregion -->

## Steps of the systematic ML process

The following steps of the systematic ML process are covered in the next main sections:

- [STEP 0: Get the dataset](#STEP-0:-Get-the-dataset)
- [STEP 1: Exploring the dataset](#STEP-1:-Exploring-the-dataset)
- [STEP 2: Prepare the dataset](#STEP-2:-Prepare-the-dataset)
- [STEP 3: Classify by support vector classifier - SVC](#STEP-3:-Classify-by-support-vector-classifier---SVC)
- [STEP 4: Evaluate the classification results - metrics](#STEP-4:-Evaluate-the-classification-results---metrics)
- [STEP 5: Select SVC kernel and vary parameters](#STEP-5:-Select-SVC-kernel-and-vary-parameters)

<!-- #region tags=[] -->
# Load globally used libraries and set plot parameters
<!-- #endregion -->

```python
import time

from IPython.display import HTML

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import seaborn as sns
%matplotlib inline
```

<!-- #region tags=[] -->
# STEP 0: Get the dataset

Since this is intended to be an introduction to the world of machine learning (ML), this step does NOT deal with the design of an application suitable for ML and the acquisition of valid measurement data.

In order to get to know the typical work steps and ML tools, the use of **well-known and well-researched data sets** is clearly **recommended**.

In the further course, the famous [Iris flower data sets](https://en.wikipedia.org/wiki/Iris_flower_data_set) will be used.
It can be downloaded on [Iris Flower Dataset | Kaggle](https://www.kaggle.com/datasets/arshid/iris-flower-dataset). Furthermore, the dataset is included in Python in the machine learning package [Scikit-learn](https://scikit-learn.org), so that users can access it without having to find a special source for it.
<!-- #endregion -->

```python
# import some data to play with
irisdata_df = pd.read_csv('./datasets/IRIS_flower_dataset_kaggle.csv')
```

<!-- #region tags=[] toc-hr-collapsed=true toc-hr-collapsed=true -->
# STEP 1: Exploring the dataset

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

<!-- #region caption="" label="fig:flower_sepal_petal" tags=[] widefigure=true -->
![Principle illustration of a flower with sepal and petal (source: [Mature_flower_diagram.svg](https://en.wikipedia.org/wiki/File:Mature_flower_diagram.svg), license: public domain)](images/Mature_flower_diagram_1024px.png)
<!-- #endregion -->

<!-- #region tags=[] -->
Here are pictures of the three different Iris species (*Iris setosa*, *Iris virginica* and *Iris versicolor*). Given the dimensions of the flower, it will be possible to predict the class of the flower.
<!-- #endregion -->

<!-- #region caption="" label="fig:Iris_setosa_virginica_versicolor" tags=[] widefigure=true -->
![Left: *Iris setosa* (source: [Irissetosa1.jpg](https://commons.wikimedia.org/wiki/File:Irissetosa1.jpg), license: public domain); middle: *Iris versicolor* (source: [Iris_versicolor_3.jpg](https://en.wikipedia.org/wiki/File:Iris_versicolor_3.jpg), license: CC-SA 3.0); right: *Iris virginica* (source: [Iris_virginica.jpg](https://en.wikipedia.org/wiki/File:Iris_virginica.jpg), license: CC-SA 2.0)
](images/Iris_images.png)
<!-- #endregion -->

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

**Boxplots** can be used to explore the data ranges in the dataset. These also provide information about **outliers**.

```python caption="Boxplots used to explore the data ranges in the Iris dataset" label="fig:boxplots_iris" tags=[] widefigure=true
sns.set_context("notebook", font_scale=1.3, rc={"lines.linewidth": 2.0})
sns.set_style("whitegrid")
#sns.set_style("white")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

fn = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
cn = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
box1 = sns.boxplot(x = 'species', y = 'sepal_length', 
                   data = irisdata_df, order = cn, ax = axs[0,0])
box2 = sns.boxplot(x = 'species', y = 'sepal_width', 
                   data = irisdata_df, order = cn, ax = axs[0,1])
box3 = sns.boxplot(x = 'species', y = 'petal_length', 
                   data = irisdata_df, order = cn, ax = axs[1,0])
box4 = sns.boxplot(x = 'species', y = 'petal_width', 
                   data = irisdata_df,  order = cn, ax = axs[1,1])

# add some spacing between subplots
fig.tight_layout(pad=2.0)

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
print("Number of rows with at least 1 NaN value: ", 
      (len(employees_df)-len(employees_df_dropped)))
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
duplicateRows = employees_df[employees_df.duplicated(
                    subset=['First Name', 'Last Login Time'])]
duplicateRows
```

```python tags=[]
# argument keep=’last’ displays the first duplicate rows instead of the last
duplicateRows = employees_df[employees_df.duplicated(
                    subset=['First Name', 'Last Login Time'], keep='last')]
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
employees_df.drop_duplicates(
    subset=['First Name', 'Last Login Time'], keep='last', inplace=True)
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
# count unique values without missing values in a column, 
# ordered descending and normalized
irisdata_df['species'].value_counts(ascending=False, dropna=False, normalize=True)
```

```python
# count unique values and missing values in a column, 
# ordered descending and not absolute values
employees_df['Team'].value_counts(ascending=False, dropna=False, normalize=False)
```

### Display Histogram

This section was inspired by: [Pandas Histogram – DataFrame.hist()](https://dataindependent.com/pandas/pandas-histogram-dataframe-hist/).

**Histograms** represent **frequency distributions** graphically. This requires the separation of the data into classes (so-called **bins**).

These classes are represented in the histogram as rectangles of equal or variable width. The height of each rectangle then represents the (relative or absolute) **frequency density**.

```python caption="Histogram for frequency distribution of the salary" tags=[] label="fig:histogram_salary" widefigure=false
employees_df.hist(column=['Salary'])
plt.show()
```

```python caption="Histogram for the frequency distribution of the salary in comparison between men and women" tags=[] label="fig:histogram_male_female" widefigure=false
employees_df.hist(column='Salary', by='Gender')
plt.show()
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

```python tags=[]
# encoding the class column
irisdata_df_enc = irisdata_df.replace({"species":  {"Iris-setosa":0,
                                                    "Iris-versicolor":1, 
                                                    "Iris-virginica":2}})
#irisdata_df_enc
```

```python
irisdata_df_enc.corr()
```

#### Correlation heatmap

Choose the color sets from [color map](https://pod.hatenablog.com/entry/2018/09/20/212527).

```python caption="Correlation heatmap to explore coherences between single variables in the iris dataset" tags=[] label="fig:correlation_heatmap" widefigure=true
# increase the size of the heatmap
plt.figure(figsize=(16, 6))

# store heatmap object in a variable to easily access it 
# when you want to include more features (such as title)
# set the range of values to be displayed on the colormap from -1 to 1,
# and set 'annotation=True' to display the correlation values on the heatmap
heatmap = sns.heatmap(irisdata_df_enc.corr(), vmin=-1, vmax=1, 
                      annot=True, cmap='PRGn_r')

# give a title to the heatmap
# 'pad=12' defines the distance of the title from the top of the heatmap
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
plt.show()
```

#### Triangle correlation heatmap

When looking at the correlation heatmaps above, you would not lose any information by **cutting** away half of it **along the diagonal** line marked by 1-s.

The **numpy** function `np.triu()` can be used to isolate the upper triangle of a matrix while turning all the values in the lower triangle into 0.

```python
import numpy as np

np.triu(np.ones_like(irisdata_df_enc.corr()))
```

Use this mask to cut the heatmap along the diagonal:

```python caption="Correlation heatmap, which was cut at its main diagonal without losing any information" tags=[] label="fig:correlation_heatmap_triangle" widefigure=true
plt.figure(figsize=(16, 6))

# define the mask to set the values in the upper triangle to 'True'
mask = np.triu(np.ones_like(irisdata_df_enc.corr(), dtype=bool))

heatmap = sns.heatmap(irisdata_df_enc.corr(), mask=mask, 
                      vmin=-1, vmax=1, annot=True, cmap='PRGn_r')

heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
plt.show()
```

As a result from the **heatmaps** we can see, that the shape of the **petals** are the **most correlationed columns** (0.96) with the **type of flowers** (species classes).

Somewhat lower correlates **sepal length** with **petal length** (0.87).

<!-- #region tags=[] -->
### Visualise data with **scatter plot**

In the following, [Seaborn](https://seaborn.pydata.org/) is applied which is a library for making statistical graphics in Python. It is built on top of matplotlib and closely integrated with pandas data structures. 

To investigate whether there are dependencies (e.g. correlations) in `irisdata_df` between individual variables in the data set, it is advisable to plot them in a **scatter plot**.
<!-- #endregion -->

```python caption="Plotting two individual variables of the iris dataset in the scatterplot to explore the relationships between these two" tags=[] label="fig:scatter_plot" widefigure=true
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


plt.title('Scatterplot of petal length and width')
plt.show()
```

### Visualise data with **pairs plot**

For systematic investigation of dependencies, all variables (each against each) are plotted in separate scatter plots.

With this so called **[pairs plot](https://vita.had.co.nz/papers/gpp.pdf)** it is possible to see both **relationships** between two variables and **distribution** of single variables.

This function will create a grid of Axes such that **each numeric variable** in `irisdata_df` will by shared in the y-axis across a single row and in the x-axis across a single column.

```python caption="Plot all individual variables of the Iris dataset in pairs plot to see both the relationships between two variables and the distribution of the individual variables" tags=[] label="fig:pairs_plot" widefigure=true
sns.set(font_scale=1.0)
sns.set_style("white")

g = sns.pairplot(irisdata_df, diag_kind="kde", hue='species', 
                 palette='Dark2', height=2.5)

g.map_lower(sns.kdeplot, levels=4, color=".2")
# y .. padding between title and plot
g.fig.suptitle('Pairs plot of the Iris dataset', y=1.05)
plt.show()
```

# STEP 2: Prepare the dataset

Through the intensive exploration of the data in Step 1 ([STEP 1: Exploring the dataset](#STEP-1:-Exploring-the-dataset)), we know that special **preparation** of the data is **not necessary**. The values are **complete** and **without gaps** and there are **no duplicates**. The values are in similar ranges, which **does not require** **normalization** of the data.

Furthermore, we know that the **classes** are very **evenly distributed** and thus bias tendencies should be avoided.

<!-- #region tags=[] toc-hr-collapsed=true -->
# STEP 3: Classify by support vector classifier - SVC

## Operating principal

> Support Vectors Classifier tries to **find the best hyperplane to separate** the different classes by maximizing the distance between sample points and the hyperplane (source: [In Depth: Parameter tuning for SVC](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769)).

The figure \ref{fig:Svm_separating_hyperplanes} shows the operating principal of the SVC algorithm: the hyperplanes *H1* till *H4* (left graphic) do separate the classes. A good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin, the lower the generalization error of the classifier (source: [Support-vector machine](https://en.wikipedia.org/wiki/Support-vector_machine)).

The right graphic shows the optimal hyperplane characterized by maximising the margin between the classes. The perpendicular distance of the closest data points to the hyperplane determines their position and orientation. These perpendicular distances are the **support vectors** of the hyperplane - this is how the algorithm got its name.
<!-- #endregion -->

<!-- #region caption="" label="fig:Svm_separating_hyperplanes" tags=[] widefigure=true -->
![Support Vectors Classifiers (SVC) separate the data points in classes by finding the best hyperplane by maximizing the margin to its support vectors](images/SVC_operatingPrinciple.png)
<!-- #endregion -->

## Split the dataset

In the next very important step, the dataset is split into **2 subsets**: a **training dataset** and a **test dataset**. As the names suggest, the training dataset is used to train the ML algorithm. The test data set is then used to check the quality of the trained ML algorithm (here the **recognition rate**). For this purpose, the **class labels** are **removed** from the training data set - after all, these are to be predicted.

Typically, the **test dataset** should contain about **20%** of the entire dataset.

```python
from sklearn.model_selection import train_test_split

X = irisdata_df.drop('species', axis=1)
y = irisdata_df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
```

For training, do not use only the variables that correlate best with each other, but all of them. 

Otherwise, the result of the prediction would be significantly worse. Maybe this is already an indication of **overfitting** of the ML model.

```python
# DO NOT USE THIS!!
X_train, X_test, y_train, y_test = train_test_split(X[['sepal_length', 
                                                       'sepal_width']], 
                                                    y, test_size = 0.20)
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

# STEP 4: Evaluate the classification results - metrics

And finally for checking the accuracy of the model, the **confusion matrix** is used for the **cross validation**.

By using the function `sklearn.metrics.confusion_matrix()` a confusion matrix of the true digit values versus the predicted digit values is plotted.

## Textual confusion matrix

```python
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
```

## Colored confusion matrix

The function `sklearn.metrics.ConfusionMatrixDisplay()` plots a colored confusion matrix.

```python caption="Checking the accuracy of the model by using the confusion matrix for cross-validation" tags=[] label="fig:confusion_matrix" widefigure=false
sns.set_style("white")

# print colored confusion matrix
cm_colored = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

cm_colored.figure_.suptitle("Colored Confusion Matrix")
cm_colored.figure_.set_figwidth(8)
cm_colored.figure_.set_figheight(7)

cm_colored.confusion_matrix

# save figure as PNG
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=150, pad_inches=5)
plt.show()
```

## Classification accuracy

```python
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, 
                             y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
```

# STEP 5: Select SVC kernel and vary parameters

This section was inspired by [In Depth: Parameter tuning for SVC](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769)

In this section, the 4 SVC parameters `kernel`, `gamma`, `C` and `degree` will be introduced one by one. Furthermore, their influence on the classification result by varying these single parameters will be shown.

**Disclaimer:** In order to show the effects of varying the individual parameters in 2D graphs, only the best correlating variables `petal_length` and `petal_width` are used to train the SVC.

## Prepare dataset

```python tags=[]
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

# import iris dataset again
irisdata_df = pd.read_csv('./datasets/IRIS_flower_dataset_kaggle.csv')

# encode the class column from class strings to integer equivalents
irisdata_df_enc = irisdata_df.replace({"species":  {"Iris-setosa":0,
                                                    "Iris-versicolor":1, 
                                                    "Iris-virginica":2}})
#irisdata_df_enc
```

### Prepare datasets for parameter variation and plotting

These datasets will  be used for parameter variation and plotting only. In particular, for later **2D plotting** of the effects of parameter variation, only **2 variables** of the iris dataset can be used.

However, as seen in the previous section, this selection is very much at the expense of detection accuracy. Therefore, it is not useful to make predictions with this subset of data - it is not necessary to divide it into a training and a test data set.

```python tags=[]
# copy only 2 feature columns
# and convert pandas dataframe to numpy array
X_plot = irisdata_df_enc[['petal_length', 'petal_width']].to_numpy(copy=True)
#X_plot = irisdata_df_enc[['sepal_length', 'sepal_width']].to_numpy(copy=True)
#X_plot
```

```python
# convert pandas dataframe to numpy array
# and get a flat 1D copy of 2D numpy array
y_plot = irisdata_df_enc[['species']].to_numpy(copy=True).flatten()
#y_plot
```

### Prepare dataset for prediction and evaluation

To **evaluate the recognition accuracy** by parameter variation, the complete iris data set with all variables must be used. To make predictions with test data, the data set is again divided into a training and a test data set.

```python
X = irisdata_df.drop('species', axis=1)
y = irisdata_df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
```

## Plotting functions

This function helps to visualize the modifications by varying the individual SVC parameters:

```python
def plotSVC(title, svc, X, y, xlabel, ylabel):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # prevent division by zero
    if x_min == 0.0:
        x_min = 0.1
    
    h = (x_max / x_min)/1000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xx.min(), xx.max())
    plt.title(title)
    plt.show()
```

This function cares for cross validation:

```python
def crossValSVC(X_train, y_train, kernel='rbf', gamma='scale', C=1.0, degree=3):
    # train the SVC
    svc = svm.SVC(kernel=kernel, 
                  gamma=gamma, 
                  C=C, 
                  degree=degree).fit(X_train, y_train)
    # calculate accuracies
    accuracies = cross_val_score(estimator = svc, X = X_train, 
                                 y = y_train, cv = 10)
    
    accuracy = accuracies.mean()*100
    return accuracy
```

This function plots the variation of the SVC parameters against the prediction accuracy to show the effect of variation and its limits regarding the phenomenon **overfitting**:

```python
def plotParamsAcc(param_list, acc_list, param_name, log_scale=False):
    fig, ax = plt.subplots(figsize=(10,6))
    title_str = 'Variation of {} parameter '.format(param_name) \
                +'and its effect to prediction accuracy'
    plt.title(title_str)
    ax.plot(param_list, accuracy_list)
    if log_scale:
        # set the X axis scale to logarithmic
        ax.set_xscale('log')
    plt.xlabel(param_name)
    plt.ylabel('accuracy [%]')
    plt.grid()
    plt.show()
```

## Vary `kernel` of SVC

The `kernel` parameter selects the type of hyperplane that is used to separate the data. Using `linear` ([linear classifier](https://en.wikipedia.org/wiki/Linear_classifier)) kernel will use a linear hyperplane (a line in the case of 2D data). The `rbf` ([radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)) and `poly` ([polynomial kernel](https://en.wikipedia.org/wiki/Polynomial_kernel)) kernel use non linear hyperplanes. The **default** is `kernel=rbf`.

```python tags=[] caption="This group of images shows the effect on the classification by the choice of the different SVC kernels ('linear', 'rbf', 'poly' and 'sigmoid')" label="fig:vary_kernels" widefigure=false
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

xlabel = 'Petal length'
ylabel = 'Petal width'

for kernel in kernels:
    svc_plot = svm.SVC(kernel=kernel).fit(X_plot, y_plot)
    accuracy = crossValSVC(X_train, y_train, kernel=kernel)
    title_str = 'kernel: \''+str(kernel)+'\', '+'Acc. prediction: {:.2f}%'.format(accuracy)
    plotSVC(title_str, svc_plot, X_plot, y_plot, xlabel, ylabel)
```

## Vary `gamma` parameter

The `gamma` parameter is used for **non linear hyperplanes**. The higher the `gamma` float value it tries to exactly fit the training data set. The **default** is `gamma='scale'`.

```python tags=[] caption="This group of images shows the effect on the classification by the variation of the parameter 'gamma' of the 'rbf' kernel" label="fig:vary_gamma_parameter" widefigure=false
gammas = [0.1, 1, 10, 100, 200]

xlabel = 'Petal length'
ylabel = 'Petal width'

for gamma in gammas:
    svc_plot = svm.SVC(kernel='rbf', gamma=gamma).fit(X_plot, y_plot)
    accuracy = crossValSVC(X_train, y_train, kernel='rbf', gamma=gamma)
    title_str = 'gamma: \''+str(gamma)+'\', ' \
                +'Acc. prediction: {:.2f}%'.format(accuracy)
    plotSVC(title_str, svc_plot, X_plot, y_plot, xlabel, ylabel)
```

Show the variation of the SVC parameter `gamma` against the **prediction accuracy**.

As we can see, increasing `gamma` leads to **overfitting** as the classifier tries to perfectly fit the training data.

```python tags=[] caption="The plot shows the variation of the SVC parameter 'gamma' against the prediction accuracy" label="fig:plot_vary_gamma" widefigure=true
gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 200]

accuracy_list = list()
for gamma in gammas:
    accuracy = crossValSVC(X_train, y_train, kernel='rbf', gamma=gamma)
    accuracy_list.append(accuracy)

plotParamsAcc(gammas, accuracy_list, 'gamma', log_scale=True)
```

## Vary `C` parameter

The `C` parameter is the **penalty** of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly. The **default** is `C=1.0`.

```python tags=[] caption="This group of images shows the effect on the classification by the variation of the parameter 'C' of the 'rbf' kernel" label="fig:vary_c_parameter" widefigure=false
cs = [0.1, 1, 5, 10, 100, 1000, 10000]

xlabel = 'Petal length'
ylabel = 'Petal width'

for c in cs:
    svc_plot = svm.SVC(kernel='rbf', C=c).fit(X_plot, y_plot)
    accuracy = crossValSVC(X_train, y_train, kernel='rbf', C=c)
    title_str = 'C: \''+str(c)+'\', ' \
                 +'Acc. prediction: {:.2f}%'.format(accuracy)
    plotSVC(title_str, svc_plot, X_plot, y_plot, xlabel, ylabel)
```

Show the variation of the SVC parameter `C` against the **prediction accuracy**.

But be careful: to high `C` values may lead to **overfitting** the training data.

```python tags=[] caption="The plot shows the variation of the SVC parameter 'C' against the prediction accuracy" label="fig:plot_vary_c" widefigure=true
cs = [0.1, 1, 5, 6, 7, 8, 10, 100, 1000, 10000]

accuracy_list = list()
for c in cs:
    accuracy = crossValSVC(X_train, y_train, kernel='rbf', C=c)
    accuracy_list.append(accuracy)

plotParamsAcc(cs, accuracy_list, 'C', log_scale=True)
```

## Vary `degree` parameter

The `degree` parameter is used when the `kernel` is set to `poly` and is ignored by all other kernels. It’s basically the **degree of the polynomial** used to find the hyperplane to split the data. The **default** is `degree=3`.

Using `degree = 1` is the same as using a `linear` kernel. Also, increasing this parameters leads to **higher training times**.

```python tags=[] caption="This group of images shows the effect on the classification by the variation of the parameter 'degree' of the 'poly' kernel" label="fig:vary_degree_parameter" widefigure=false
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

xlabel = 'Petal length'
ylabel = 'Petal width'

for degree in degrees:
    svc_plot = svm.SVC(kernel='poly', degree=degree).fit(X_plot, y_plot)
    accuracy = crossValSVC(X_train, y_train, kernel='poly', degree=degree)
    title_str = 'degree: \''+str(degree)+'\', ' \
                 +'Acc. prediction: {:.2f}%'.format(accuracy)
    plotSVC(title_str, svc_plot, X_plot, y_plot, xlabel, ylabel)
```

Show the variation of the SVC parameter `degree` against the **prediction accuracy**.

As we can see, increasing the `degree` of the polynomial hyperplane leads to **overfitting** the training data.

```python tags=[] caption="The plot shows the variation of the SVC parameter 'degree' against the prediction accuracy" label="fig:plot_vary_degree" widefigure=true
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

accuracy_list = list()
for degree in degrees:
    accuracy = crossValSVC(X_train, y_train, kernel='poly', degree=degree)
    accuracy_list.append(accuracy)

plotParamsAcc(degrees, accuracy_list, 'degree', log_scale=False)
```

```python

```
