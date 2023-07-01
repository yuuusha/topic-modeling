#используемые библиотеки
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import gensim
    from sklearn.feature_extraction import text
    from gensim.models import LdaMulticore
    from gensim.corpora import Dictionary
    from sklearn.feature_extraction.text import CountVectorizer
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from sklearn.metrics import confusion_matrix, classification_report

    #загрузка данных
    papers = pd.read_csv("temp/nlp/Train.csv")

    #подготовка данных и векторизация
    vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                token_pattern='(?u)\\b\\w\\w\\w+\\b')
    X = vect.fit_transform(papers.ABSTRACT)

    #мешок слов и список идентификаторов
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    id_map = dict((v, k) for k, v in vect.vocabulary_.items())

    #построение модели
    ldamodel = gensim.models.LdaMulticore(corpus=corpus, id2word=id_map, passes=2, random_state=5, 
                num_topics=4, workers=2)

    #вывод слов по темам
    for idx, topic in ldamodel.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic))
        print("\n")

    #вывод облака слов
    for t in range(ldamodel.num_topics):
        plt.figure()
        plt.imshow(WordCloud(background_color="white").fit_words(dict(ldamodel.show_topic(t, 100))))
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.show()

    #загрузка тестовых данных
    tests = pd.read_csv("temp/nlp/Test.csv")

    #функция для разметки по выявленным темам
    def target(x):
        if tests["Computer Science"][x] == 1:
            return 1
        elif tests["Mathematics"][x] == 1:
            return 2
        elif tests["Physics"][x] == 1:
            return 3
        elif tests["Statistics"][x] == 1:
            return 0

    #разметка
    tests["target"] = [target(x) for x in range(tests.shape[0])]

    #функция предсказания
    def topic_prediction(my_document):
        string_input = [my_document]
        X = vect.transform(string_input)
        corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
        output = list(ldamodel[corpus])[0]
        topics = sorted(output,key=lambda x:x[1],reverse=True)
        return topics[0][0]

    #предсказание
    tests["prediction"] = [topic_prediction(test_text) for test_text in tests.ABSTRACT]
    
    #сравнение двух колонок - оценка точности
    print(confusion_matrix(tests['target'], tests['prediction']))
    print(classification_report(tests['target'], tests['prediction']))