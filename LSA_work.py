#используемые библиотеки
import pandas as pd
import gensim
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import TruncatedSVD 
import umap.umap_ as umap
from sklearn.metrics import confusion_matrix, classification_report

#загрузка данных
papers = pd.read_csv("temp/nlp/Train.csv")

#подготовка данных и векторизация
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', token_pattern='(?u)\\b\\w\\w\\w+\\b')
X = vect.fit_transform(papers.ABSTRACT)

#мешок слов и список идентификаторов
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

#построение модели
lsamodel = gensim.models.LsiModel(corpus, id2word=id_map, num_topics=4)

#вывод слов по темам
print(lsamodel.print_topics(num_topics=4, num_words=10))

#функция для разметки по выявленным темам
def target(x, table):
    if table["Computer Science"][x] == 1:
        return 0
    if table["Mathematics"][x] == 1:
        return 2
    elif table["Physics"][x] == 1:
        return 3
    elif table["Statistics"][x] == 1:
        return 1

#разметка
papers["target"] = [target(x, papers) for x in range(papers.shape[0])]

#вывод облака слов
for t in range(lsamodel.num_topics):
    plt.figure()
    plt.imshow(WordCloud(background_color="white").fit_words(dict(lsamodel.show_topic(t, 100))))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()

#модель SVD для перехода в семантическое пространство
svd_model = TruncatedSVD(n_components=4, algorithm='randomized', n_iter=100, random_state=122) 
svd_model.fit(X)
X_topics = svd_model.fit_transform(X) 

#визуализация семантического пространства
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X) 
plt.figure(figsize=(7,5)) 
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
c = papers.target,
s = 10, # size 
edgecolor='none')
plt.colorbar()
plt.show()

#загрузка тестовых данных
tests = pd.read_csv("temp/nlp/Test.csv")

#разметка
tests["target"] = [target(x, tests) for x in range(tests.shape[0])]

#функция предсказания
def topic_prediction(my_document):
    string_input = [my_document]
    X = vect.transform(string_input)
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    output = list(lsamodel[corpus])[0]
    topics = sorted(output,key=lambda x:x[1],reverse=True)
    return topics[0][0]

#предсказание
tests["prediction"] = [topic_prediction(test_text) for test_text in tests.ABSTRACT]

#сравнение двух колонок - оценка точности
print(confusion_matrix(tests['target'], tests['prediction']))
print(classification_report(tests['target'], tests['prediction']))