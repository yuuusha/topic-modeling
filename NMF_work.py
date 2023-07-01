#используемые библиотеки
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, classification_report

#загрузка данных
papers = pd.read_csv("temp/nlp/Train.csv")

#подготовка данных и векторизация
text_data = [x for x in papers.ABSTRACT]
vectorizer = CountVectorizer(max_features=1500, min_df=10, stop_words='english', token_pattern='(?u)\\b\\w\\w\\w+\\b')
X = vectorizer.fit_transform(text_data)
words = np.array(vectorizer.get_feature_names_out())
 
#построение модели NMF
nmf = NMF(n_components=4, solver="mu")
W = nmf.fit_transform(X)
H = nmf.components_

#вывод облака слов
for t, topic in enumerate(H):
    x = dict(zip(words, topic))
    #myDict = {key:val for key, val in x.items() if val != 0}
    a = dict(sorted(x.items(), key= lambda item: item[1], reverse=True))
    lst = []
    for key, value in a.items():
        lst.append([key, value])
    plt.figure()
    plt.imshow(WordCloud(background_color="white").fit_words(dict(lst[0:100])))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()
    
    #print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in words[topic.argsort()[-10:]]])

#загрузка тестовых данных
tests = pd.read_csv("temp/nlp/Test.csv")

#функция для разметки по выявленным темам
def target(x, table):
    if table["Computer Science"][x] == 1:
        return 2
    elif table["Mathematics"][x] == 1:
        return 1
    elif table["Physics"][x] == 1:
        return 0
    elif table["Statistics"][x] == 1:
        return 3

#разметка
tests["target"] = [target(x, tests) for x in range(tests.shape[0])]

#подготовка новых данных и векторизация
new_text = [x for x in tests.ABSTRACT]
vect_new = vectorizer.transform(new_text)
X_new = nmf.transform(vect_new)

#предсказание
predicted_topics = [np.argsort(each)[::-1][0] for each in X_new]
tests['prediction'] = predicted_topics

#сравнение двух колонок - оценка точности
print(confusion_matrix(tests['target'], tests['prediction']))
print(classification_report(tests['target'], tests['prediction']))