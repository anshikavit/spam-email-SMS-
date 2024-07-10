import numpy as np
import pandas as pd

encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
file_path = r"C:\New folder\spam\spam.csv"  # Use a raw string or escape the backslashes
 
for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"File successfully read with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Failed to read with encoding: {encoding}")
        continue

if 'df' in locals():
    print("CSV file has been successfully loaded.")
    print(df.sample(5))
    print(df.shape)
else:
    print("All encoding attempts failed")

df.info()
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.sample(5)
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
print(df.sample(5))
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
df.head()
df.isnull().sum()
print("Duplicated values are:",df.duplicated().sum())
df=df.drop_duplicates(keep='first')
print("Duplicated values after removing duplicates once are:",df.duplicated().sum())
print(df.shape)

#eda(data exploration)
print(df['target'].value_counts())

import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")

#natural tool kit 
import nltk
nltk.download('punkt')
#character count
df['num_characters']=df['text'].apply(len)
df.head()
#text count
df['num_word']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df.head()
df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head()
df[['num_characters','num_word','num_sentences']].describe()
#targeting ham
df[df['target']==0][['num_characters','num_word','num_sentences']].describe()
#targetting spam
df[df['target']==1][['num_characters','num_word','num_sentences']].describe()


#Data Preproccessing
#change all the characters into lower case letters
#Tokenization- breaking down sentences into words
#Removing special characters
#removing stop words and punctuations
#Stemming- when you have a word into different forms, all will be changed to the same form 

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

transformed_text = transform_text("I'm gonna be home tomorrow and I don't want to talk about this stuff.")
print(transformed_text)

df['text'][10]
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
ps.stem('walking')
df['transformed_text']= df['text'].apply(transform_text)
df.head()

spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
len(spam_corpus)
print(len(spam_corpus))

ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
len(ham_corpus)
print(len(ham_corpus))

#Building the model

from sklearn.feature_extraction.text  import CountVectorizer, TfidfVectorizer
cv=CountVectorizer()
tfidf= TfidfVectorizer(max_features=3000)
X= tfidf.fit_transform(df['transformed_text']).toarray()
y=df['target'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score
mnb=MultinomialNB()
bnb=BernoulliNB()

mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

#tfidf --> MNB

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc=SVC(kernel='sigmoid',gamma=1.0)
etc=ExtraTreesClassifier(n_estimators=50,random_state=2)

clfs= {
    'SVC':svc,
    'ETC':etc
}
def train_classifier(clf,X_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)

    return accuracy,precision
train_classifier(svc,X_train,y_train,y_test)
accuracy_score=[]
precision_score=[]

for name,clf in clfs.items():
    current_accuracy,current_precision=train_classifier(clf,X_train,y_train,X_test,y_test)

    print("For",name)
    print("Accuracy",current_accuracy)
    print("Precision" , current_precision)

    accuracy_score.append(current_accuracy)
    precision_score.append(current_precision)

#model improve
df=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_score,'Precision_max_ft_3000':precision_score}).sort_values
new_df=performance_df.merge(temp_df,on='Algorithm')
new_df_scaled=new_df.merge(temp_df,on='Algorithm')

temp_df=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_score,'Precision_max_ft_3000':precision_score}).sort_values
new_df_scaled.merge(temp_df,on='Algorithm')

#voting classifier
svc=SVC(kernel='sigmoid',gemma=1.0,probability=True)
mnb=MultinomialNB()
etc=ExtraTreesClassifier(n_estimators=50, random_state=2)
from sklearn.ensemble import VotingClassifier
voting=VotingClassifier(estimators=[('svm',svc),('nb',mnb),('et',etc)],voting='soft')
voting.fit(X_train,y_train)
y_pred=voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


#Applying stacking
estimators=[('svm',svc),('nb',mnb),('et',etc)]
final_estimator=RandomForestClassifier()
from sklearn.ensemble import StackingClassifier
clf=StackingClassifier(estimators=estimators,final_estimator=final_estimator)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#sample text data and correspomding lables
X_train=["Sample text 1","Sample text 2","Sample text 3"]
y_train=[0, 1, 0]
#create and train t he TF-IDF vectorizer
tfidf=TfidfVectorizer(lowercase=True, stop_words='english')
X_train_tfidf=tfidf.fit_transform(X_train)

#create and train the naive bayes classifier
mnb=MultinomialNB()
mnb.fit(X_train_tfidf,y_train)

#Save the trained TF-IDF vectorizer and naive bayes mmodel o files
with open('vectorizer.pkl','wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

with open('model.pkl','wb') as model_file:
    pickle.dump(mnb,model_file)
