import nltk
stopwords = nltk.corpus.stopwords.words('english')

def cleanDataset(sentence):
  data = list(set(sentence.split(" ")) - set(stopwords))
  return " ".join(data)