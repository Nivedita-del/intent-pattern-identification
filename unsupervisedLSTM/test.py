import nltk
nltk.download("books")
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import wikipedia

#document = 'Today the Netherlands celebrates King\'s Day. To honor this tradition, the Dutch embassy in San Francisco invited me to'
document = "hi is nive ?"
#document = 'Whether you\'re new to programming or an experienced developer, it\'s easy to learn and use Python.'

sentences = nltk.sent_tokenize(document)

data = []
for sent in sentences:
    data = data + nltk.pos_tag(nltk.word_tokenize(sent))

print(data)
for word in data:
    if('WRB' in word[1] or 'WP' in word[1]):
        print(word)
        for word in data:
            if(word[1]=='NNP'):
                print(wikipedia.summary(word[0]))