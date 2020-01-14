import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/Users/mac/Downloads/titanic/train.csv")
print (df.shape)
print (df.count)


df.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

"""Trying to display both the bars together"""
plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

"""Drawing scatter plot between the age and the number of dead people """
plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age,alpha= 0.1)
plt.title("Age wrt Survived")

"""Drawing scatter plot between the class and the number of people in those classes """
plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Class")


"""Here we are trying to correlate the age vs the calss ticket of the person so we use Kernel density for this """
plt.subplot2grid((2,3),(1,0),colspan=2)
for x in [1,2,3]:
    df.Age[df.Pclass == x].plot(kind="kde")
plt.title("Age and Class Correlation")
plt.legend(("1st","2nd","3rd"))

"""Graph showcasing where people boarded the ship"""
plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Embarked")
plt.show()


"""Trying to plot the graphs for the genders"""

female_color = "#FA0000"

plt.subplot2grid((3,4),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived vs Died")

"""Plotting the the graphs for the male survivors"""
plt.subplot2grid((3,4),(0,1))
df.Survived[df.Sex == "male"].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Male Survivors")

"""Plotting the the graphs for the female survivors"""
plt.subplot2grid((3,4),(0,2))
df.Survived[df.Sex == "female"].value_counts(normalize=True).plot(kind="bar",alpha=0.5, color=female_color)
plt.title("Female Survivors")

"""Plotting the the graphs for the male versus the female survivors"""
plt.subplot2grid((3,4),(0,3))
df.Sex[df.Survived == 1].value_counts(normalize=True).plot(kind="bar",alpha=0.5, color=[female_color,'b'])
plt.title("Male versus Female Survivors")

"""Using the class to show the survival rate using Kernel Density Estimate""" 
plt.subplot2grid((3,4),(1,0),colspan=4)
for x in [1,2,3]:
    df.Survived [df.Pclass == x].plot(kind="kde")
plt.title("Survived wrt Class Correlation")
plt.legend(("1st","2nd","3rd"))


"Trying to see how the rich vs poor died"

plt.subplot2grid((3,4),(2,0))
df.Survived[(df.Sex=="male") & (df.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Rich Men Survived")

plt.subplot2grid((3,4),(2,1))
df.Survived[(df.Sex=="male") & (df.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Poor Men Survived")

plt.subplot2grid((3,4),(2,2))
df.Survived[(df.Sex=="female") & (df.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5, color= female_color)
plt.title("Rich Women Survived")

plt.subplot2grid((3,4),(2,3))
df.Survived[(df.Sex=="female") & (df.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color= female_color)
plt.title("Poor Women Survived")

plt.show()



"""Now we start implementing machine learning algorithm"""
"""********* trying to manipulate the data in the dataframe"""
df1 = pd.read_csv("/Users/mac/Downloads/titanic/df1.csv")

"""adding a new column to our data set"""

df1["Hyp"]=0
df1.loc[df1.Sex=="female","Hyp"]=1

df1["Result"]=0
df1.loc[df1.Survived == df1["Hyp"], "Result"]=1

print (df1["Result"].value_counts(normalize= True))

"""*******finished the manipulation of the data"""


df1["Fare"]= df1["Fare"].fillna(df1["Fare"].dropna().median())
df1["Age"]= df1["Age"].fillna(df1["Age"].dropna().median())

print(df1["Fare"])


"""Cleaning up the data and filling up the void values with the median"""
df1["Fare"]= df1["Fare"].fillna(df1["Fare"].dropna().median())
df1["Age"]= df1["Age"].fillna(df1["Age"].dropna().median())

"""Assigning the numeric values to both male and female """

df1.loc[df1["Sex"]=="male", "Sex"]=0
df1.loc[df1["Sex"]=="female", "Sex"]=1


"""assigning the numeric values to all the three locations where the poeple boarded on the ship"""

df1["Embarked"] = df1["Embarked"].fillna("S")
df1.loc[df1["Embarked"]== "S", "Embarked"] = 0
df1.loc[df1["Embarked"]== "C", "Embarked"] = 1
df1.loc[df1["Embarked"]== "Q", "Embarked"] = 2



"""Now we start the implementation of the Machine learning algorithm from here"""

"""import utils"""
from sklearn import linear_model

"""utils.clean_data(df1)"""

"""We clearly define target and features for the machine learning algorithm"""

target = df1.Survived.values
features = df1[["Pclass","Age","Sex","SibSp","Parch"]].values


"""Now we create a classifier"""

classifier = linear_model.LogisticRegression()
classifier_final = classifier.fit(features, target)

print (classifier_final.score(features, target))



"""We clearly define target and features for the machine learning algorithm"""

target = df1.Survived.values
features = df1[["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]].values

"""Now we create a classifier"""

classifier = linear_model.LogisticRegression()
classifier_final = classifier.fit(features, target)

print (classifier_final.score(features, target))


"""Now we start the implementation of the Machine learning Logistic Regression algorithm from here"""

"""import utils"""
from sklearn import linear_model, preprocessing

"""We clearly define target and features for the machine learning algorithm"""

target = df1.Survived.values
feature_names= ["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]
features = df1[feature_names].values

"""Now we create a classifier"""

classifier = linear_model.LogisticRegression()
classifier_final = classifier.fit(features, target)

print (classifier_final.score(features, target))

"""NOw here we will take our linear data and combine it into 2nd degree polynomial"""

poly = preprocessing.PolynomialFeatures(degree=2 )
poly_features = poly.fit_transform(features)

classifier = linear_model.LogisticRegression()
classifier_final = classifier.fit(poly_features, target)

print (classifier_final.score(poly_features, target))



 """Now we start the implementation of the Machine learning Decision Tree algorithm from here"""
    
from sklearn import tree , model_selection
    
target = df1.Survived.values
feature_names= ["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]
features = df1[feature_names].values

decision_tree = tree.DecisionTreeClassifier(random_state= 1)
decision_tree_final = decision_tree.fit(features,target)

print (decision_tree_final.score(features, target))

scores = model_selection.cross_val_score(decision_tree,features,target,scoring ='accuracy', cv=50)
print (scores)
print (scores.mean())


generalized_tree = tree.DecisionTreeClassifier(
    random_state= 1,
    max_depth =7,
    min_samples_split=2
)

generalized_tree_final = decision_tree.fit(features,target)

scores = model_selection.cross_val_score(decision_tree,features,target,scoring ='accuracy', cv=50)
print (scores)
print (scores.mean())


generalized_tree = tree.DecisionTreeClassifier(
    random_state= 1,
    max_depth =7,
    min_samples_split=2
)

generalized_tree_final = decision_tree.fit(features,target)

scores = model_selection.cross_val_score(decision_tree,features,target,scoring ='accuracy', cv=50)
print (scores)
print (scores.mean())



  """trying to create a visual depiction of the tree which the algorithm has based it's decision on"""
    
tree.export_graphviz(generalized_tree_final, feature_names= feature_names, out_file="tree.dot")

"""To convert a dot file into the png file you can use the below mentioned command or you can do the conversion online fromdot to png converter"
""
"""dot -Tpng tree.dot > tree.png"""


