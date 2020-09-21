import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

TAG_MAP = {
    "NOUN": {"pos": "NOUN"},  # noun
    "VERB": {"pos": "VERB"},  # verb
    "ADJ": {"pos": "ADJ"},  # adjective
    "ADP": {"pos": "ADP"},  # adposition
    "ADV": {"pos": "ADV"},  # adverb
    "AUX": {"pos": "AUX"},  # auxiliary verb
    "CONJ": {"pos": "CONJ"},  # coordinating conjunction
    "DET": {"pos": "DET"},  # determiner
    "INTJ": {"pos": "INTJ"},  # interjection
    "NUM": {"pos": "NUM"},  # numeral
    "PART": {"pos": "PART"},  # particle
    "PRON": {"pos": "PRON"},  # pronoun
    "PROPN": {"pos": "PROPN"},  # proper noun
    "PUNCT": {"pos": "PUNCT"},  # punctuation
    "SCONJ": {"pos": "SCONJ"},  # subordinating conjunction
    "SYM": {"pos": "SYM"}  # symbol
}

TRAIN_DATA = [
    ("Hi hello hola namaste ola welcome .", {"tags": ["INTJ", "INTJ", "INTJ", "INTJ", "INTJ", "INTJ", "SYM"]}),
    ("I like green eggs", {"tags": ["NOUN", "VERB", "ADJ", "NOUN"]}),
    ("Eat blue ham", {"tags": ["VERB", "ADJ", "NOUN"]}),
    ("Cat sat on a Mat", {"tags": ["NOUN", "VERB", "ADV", "DET", "NOUN"]}),
    ("A treat for fans of Bob Marley", {"tags": ["DET", "VERB", "ADP", "NOUN", "ADP", "NOUN", "NOUN"]}),
    ("How are you ?", {"tags": ["ADV", "VERB", "PRON", "SYM"]}),
    ("I am angry sad depressed lonely ,", {"tags": ["NOUN", "VERB", "ADJ", "ADJ", "ADJ", "ADJ", "SYM"]})
]

# -------------------
lang = "en"
output_dir = "./"
n_iter = 100
# -------------------

nlp = spacy.blank(lang)

tagger = nlp.create_pipe("tagger")
for tag, values in TAG_MAP.items():
    tagger.add_label(tag, values)
nlp.add_pipe(tagger)

optimizer = nlp.begin_training()
for i in range(n_iter):
    random.shuffle(TRAIN_DATA)
    losses = {}

    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(texts, annotations, sgd=optimizer, losses=losses)
    print("Losses", losses)

test_text = "I am sad, angry and depressed"
doc = nlp(test_text)
print("Tags", [(t.text, t.tag_, t.pos_) for t in doc])

if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc = nlp2(test_text)
    print("Tags", [(t.text, t.tag_, t.pos_) for t in doc])