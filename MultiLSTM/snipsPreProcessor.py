import json
import os
import re
import numpy as np
import pickle
from collections import Counter
from preprocessing import CharacterIndexer, SlotIndexer, IntentIndexer
from gensim.models import Word2Vec

# # fix bad line in the PlayMusic file
# # then manually rename
# def decode(s, encoding="utf8", errors="ignore"):
#     return s.decode(encoding=encoding, errors=errors)

# raw_json_file = open('data/snips/train/train_PlayMusic_full.json', 'rb')
# raw_json = decode(bytes(raw_json_file.read()))
# all_str = raw_json[ raw_json.find("{"): ]
# all_obj = json.loads(all_str)
# with open('data/snips/train/train_PlayMusic_full_fixed.json', 'w') as outfile:
#     json.dump(all_obj, outfile)


train_sents, train_tags, train_intents = [], [], []
path = 'data/snips/train'
for filename in os.listdir(path):
    if 'json' in filename:
        with open(path + '/' + filename, encoding='utf8') as json_file:
            intent = filename.split('_')[1]
            print(filename, intent)
    #         try:
            data = json.load(json_file)
            data = data[intent]
            for sent in data:
                s, t = [], []
                for dct in sent['data']:
                    if 'entity' in dct.keys():
                        t.append(dct['entity'])
                        s.append(dct['text'])
                    else:
                        t.append("NONE")
                        s.append(dct['text'])
                train_sents.append(s)
                train_tags.append(t)
                train_intents.append(intent)
#         except UnicodeDecodeError:
#             pass

len(train_sents), len(train_tags), len(train_intents)

Counter(train_intents).most_common()

val_sents, val_tags, val_intents = [], [], []
path = 'data/snips/validate'
for filename in os.listdir(path):
    if 'json' in filename:
        with open(path + '/' + filename) as json_file:
            intent = filename.split('_')[1]
            intent = intent.split('.')[0]
            print(filename, intent)
#             try:
            data = json.load(json_file)
            data = data[intent]
            for sent in data:
                s, t = [], []
                for dct in sent['data']:
                    if 'entity' in dct.keys():
                        t.append(dct['entity'])
                        s.append(dct['text'])
                    else:
                        t.append("NONE")
                        s.append(dct['text'])
                val_sents.append(s)
                val_tags.append(t)
                val_intents.append(intent)
#             except UnicodeDecodeError:
#                 pass

len(val_sents), len(val_tags), len(val_intents)


Counter(val_intents).most_common()

print(train_sents[0])
print(train_tags[0])
print(train_intents[0])

# preprocess sentences
def cleanup(sentlist, taglist):
    newsents = []
    newtags  = []
    for idx, sent in enumerate(sentlist):
        newsent, newtag = [], []
        for jdx, phrase in  enumerate(sent):
            for c in ['.', ',', '!', '?', ]:
                phrase = phrase.replace(c, '')
            tt = phrase.split()
            for kdx, t in enumerate(tt):
                # digit replacement
                # if t.isdigit():
                #     newsent.append(digits.get(t, '##'))
                # else:
                newsent.append(t.lower())
                newtag.append(taglist[idx][jdx])
        newsents.append(newsent)
        newtags.append(newtag)
    return newsents, newtags

train_sents, train_tags = cleanup(train_sents, train_tags)

val_sents, val_tags = cleanup(val_sents, val_tags)


print(train_sents[0])
print(train_tags[0])
print(train_intents[0])

len(list(set([t for s in train_tags for t in s])))

Counter(list([t for s in train_tags for t in s])).most_common(10)

# train and save model
model = Word2Vec(train_sents, size=200, min_count=1, window=3, workers=3, iter=5)
model.save('model/snips_w2v.gensimmodel')
print('training done!')
# get model vocabulary
vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])

# test
model.wv.most_similar('hear')

vocab = list(set([w for s in train_sents for w in s]))
len(vocab)

VOCABSIZE = 10000

lens = [len(s) for s in train_sents]
_MAXLEN = int(np.round(np.mean(lens) + 2*np.std(lens)))
_MAXLEN

sum([1 if lens[i] <= _MAXLEN else 0 for i in range(len(lens))])/len(lens)

sent_indexer = CharacterIndexer(max_sent_len=15, max_word_mode='std', max_word_vocab=10000)

sent_indexer.fit(train_sents, verbose=False)

# transform the sentence data
trn_text_idx, trn_char_idx = sent_indexer.transform(train_sents)
tst_text_idx, tst_char_idx = sent_indexer.transform(val_sents)
trn_text_idx.shape, trn_char_idx.shape, tst_text_idx.shape, tst_char_idx.shape

# transform the slot data
slot_indexer = SlotIndexer(max_len=15)
slot_indexer.fit(train_tags)
trn_slot_idx = slot_indexer.transform(train_tags)
tst_slot_idx = slot_indexer.transform(val_tags)
trn_slot_idx.shape, tst_slot_idx.shape

# transform the intent data
int_indexer = IntentIndexer()
int_indexer.fit(train_intents)
trn_int_idx = int_indexer.transform(train_intents)
tst_int_idx = int_indexer.transform(val_intents)
trn_int_idx.shape, tst_int_idx.shape

pickle.dump(train_sents,   open('data/snips/train_sents.pkl', 'wb'))
pickle.dump(train_tags,    open('data/snips/train_slots.pkl', 'wb'))
pickle.dump(train_intents, open('data/snips/train_intents.pkl', 'wb'))
pickle.dump(val_sents,   open('data/snips/test_sents.pkl', 'wb'))
pickle.dump(val_tags,    open('data/snips/test_slots.pkl', 'wb'))
pickle.dump(val_intents, open('data/snips/test_intents.pkl', 'wb'))

np.save('encoded/snips_x_trn_text.npy', trn_text_idx)
np.save('encoded/snips_x_tst_text.npy', tst_text_idx)

np.save('encoded/snips_x_trn_char.npy', trn_char_idx)
np.save('encoded/snips_x_tst_char.npy', tst_char_idx)

np.save('encoded/snips_y_trn_slot.npy', trn_slot_idx)
np.save('encoded/snips_y_tst_slot.npy', tst_slot_idx)

np.save('encoded/snips_y_trn_ints.npy', trn_int_idx)
np.save('encoded/snips_y_tst_ints.npy', tst_int_idx)

pickle.dump(sent_indexer, open("encoded/snips_sent_indexer.pkl", "wb"))
pickle.dump(slot_indexer, open("encoded/snips_slot_indexer.pkl", "wb"))
pickle.dump(int_indexer,  open("encoded/snips_int_indexer.pkl", "wb"))

pickle.dump(vocab,  open("model/snips_w2v_vocab.pkl", "wb"))

