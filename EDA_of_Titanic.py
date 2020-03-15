import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.getLogger().setLevel(logging.CRITICAL)
from warnings import simplefilter
simplefilter(action = 'ignore', category = FutureWarning)
sns.set()
#import the data
data = pd.read_csv('titanic data.csv')
#explore the data
data.head()
data.info()
data.shape
#drop the Cabin column because the nil vacancies is more than 60%

data = data.drop('Cabin', axis = 1)
data.shape
#percentage of survival

surv_percent = data.groupby('Survived').size()
print(surv_percent)
surv_percent = pd.DataFrame(surv_percent)
plot = surv_percent.plot.pie(subplots = True)
plt.savefig('Quick glance at the ratio of the dead to the survivors.png')
#To observe the frequency of the Ages.

data['Age'].plot.hist()
plt.savefig('Range of age of people who travelled.png')
#the Age has an imporant tole to play, so we fill in the missing value, preferably with the younguns 

data['Age'] = data['Age'].fillna(-0.5) 
#bucketing the Age categories for a better view of the age ranges that survived.

Cut_points = [-1,0, 5, 12, 18, 35, 60, 100]
labels_names = ['Missing', 'Infant', 'Child', 'Teenager', 'Young adult', 'Adult','Senior adults' ]
    
data['Age_categories'] = pd.cut(data['Age'], Cut_points, labels = labels_names)
    
    
age_cat_pivot = data.pivot_table(index = 'Age_categories', values = 'Survived')
print(age_cat_pivot)
age_cat_pivot.plot.hist()
#survival rate as regards to gender

surv_gen = data.pivot_table(index = 'Sex', values = 'Survived')
print(surv_gen)
Percentage_of_survival = (surv_gen/(surv_gen.sum())) *100
print(Percentage_of_survival)
Percentage_of_survival.plot.pie(subplots = True)
plt.savefig('percentage of survival relative to gender.png')
pop_val = data['Sex'].value_counts()
print(pop_val)
percentages= (pop_val/(pop_val.sum())) *100
print(percentages)
percentages.plot.pie(subplots = True)
plt.savefig('The percentageof male to female onboard.png')
#population of the cabins

pop_pclass = data['Pclass'].value_counts()
print(pop_pclass)
pop_pclass_percent = (pop_pclass/(pop_pclass.sum())) *100
print(pop_pclass_percent)
pop_pclass_percent.plot.barh()
plt.savefig('class population percentage.png')
#survival rate according to the cabin class.

surv_pclass = data.pivot_table(index = 'Pclass', values ='Survived')
print(surv_pclass)
surv_pclass_percent = (surv_pclass/(surv_pclass.sum())) *100
print(surv_pclass_percent)
surv_pclass_percent.plot.barh()
plt.savefig('percentage of survival per class.png')

#where people boarded the most

pop_boarding_loc = data.groupby('Embarked').size()
print(pop_boarding_loc)
pop_boarding_loc.plot.bar()
plt.savefig('size of population where people boarded.png')
data.plot.scatter('Pclass', 'Fare')
plt.savefig('data fare.png')
data['Fare'].max()
#max fare is 512.3292
data[np.isclose(data['Fare'],512.3292)]
