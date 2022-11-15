import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tf = TfidfVectorizer()
mnb = MultinomialNB()
test = input("Enter:")
test_data = []
test_data.append(test)
df = pd.read_csv("SMSSpamCollection", sep = "\t", names = ["Status", "Message"])
df.loc[df["Status"] == "ham", "Status"] = 1
df.loc[df["Status"] == "spam", "Status"] = 0
df_x=df['Message']
df_y=df['Status']
x_traintf= tf.fit_transform(df_x).toarray()
y_train= df_y.astype('int')
x_train,x_test,y_train,y_test = train_test_split(x_traintf,y_train)
X = tf.transform(test_data).toarray()
mnb.fit(x_train, y_train)
pred= mnb.predict(X)
if pred==1:
    print ("not spam")
else:
    print("spam")
pred = mnb.predict(x_test)
sc = accuracy_score(y_test,pred)

print("Accuracy of the Model: {} %".format(sc*100))