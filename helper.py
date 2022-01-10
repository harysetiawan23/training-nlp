from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
stopwords = nltk.corpus.stopwords.words('english')

cleaned_text = []

def parseStringToVector(sentence):
  data = list(set(sentence.split(" ")) - set(stopwords))
  cleaned_text.append(" ".join(data))


  TFIDF = TfidfVectorizer(analyzer='word',stop_words='english',ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)
  model = TFIDF.fit_transform(cleaned_text)

  word = TFIDF.get_feature_names()
  return model.toarray(),word


