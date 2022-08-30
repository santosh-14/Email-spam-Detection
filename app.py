import streamlit as st
import string
from nltk.corpus import stopwords
import nltk
import pickle
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()


cv=pickle.load(open("vectorizer.pkl","rb"))
mnb=pickle.load(open("model.pkl","rb"))

def transform_func(x):
    x=x.lower()
    #for converting text to list for further transformation
    x=nltk.word_tokenize(x)
    
    # fordropping words other than alpganumeric
    l=[]
    for i in x:
        if i.isalnum():
            l.append(i)
    # for removing stopwords
    x=l[:]
    l.clear()
    for i in x:
        if i not in string.punctuation and i not in stopwords.words("english"):
            l.append(i)
    # for stemming
    x=l[:]
    l.clear()
    for i in x:
        l.append(ps.stem(i))
    return " ".join(l)



st.title("E-mail Classifier")
ip=st.text_area("Enter email here to check whether it is spam or not")

if st.button("predict"):
    transformed=transform_func(ip)
    vectorize=cv.transform([transformed])
    res=mnb.predict(vectorize)

    if res==1:
        st.header("spam")
    else:
        st.header("not spam")
