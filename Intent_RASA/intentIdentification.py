import rasa_nlu
import rasa_core
from rasa_nlu.model import Metadata, Interpreter

import pandas as pd

df = pd.read_csv('inp.csv', sep='\t', engine='python')


def load_model_tag():
    list1 = []
    interpreter = Interpreter.load('models/current/nlu_model')
    for i in range(df.shape[0]):
        x = interpreter.parse(df.Post[i])
        passed = {key: value for key, value in x.items() if key == 'intent'}
        tag = [d['name'] for d in passed.values() if 'name' in d]
        list1.append(tag)

    return (list1)

def load_model_confidence():
    list2 = []
    interpreter = Interpreter.load('models/current/nlu_model')
    for i in range(df.shape[0]):
        x = interpreter.parse(df.Post[i])
        passed = {key: value for key, value in x.items() if key == 'intent'}
        confidence = [d['confidence'] for d in passed.values() if 'confidence' in d]

        list2.append(confidence)

    return (list2)


def load_2nd_tag():
    list3 = []
    interpreter = Interpreter.load('models/current/nlu_model')
    for i in range(df.shape[0]):
        x = interpreter.parse(df.Post[i])
        v1 = {keys: values for keys, values in x.items() if keys == "intent_ranking"}
        v1 = v1['intent_ranking']
        nd_tag = [d['name'] for d in v1 if 'name' in d]
        nd_tag = nd_tag[1]
        # nd_confidence=round(nd_confidence,2)
        list3.append(nd_tag)

    return (list3)

def load_2nd_confidence():
    list4 = []
    interpreter = Interpreter.load('models/current/nlu_model')
    for i in range(df.shape[0]):
        x = interpreter.parse(df.Post[i])
        v1 = {keys: values for keys, values in x.items() if keys == "intent_ranking"}
        v1 = v1['intent_ranking']
        nd_confidence = [d['confidence'] for d in v1 if 'confidence' in d]
        nd_confidence = nd_confidence[1]
        nd_confidence = str(nd_confidence)
        list4.append(nd_confidence)

    return (list4)



model_tag_res = load_model_tag()
model_con_res = load_model_confidence()
model_2nd_tag_res = load_2nd_tag()
model_2nd_confidence = load_2nd_confidence()

model_tag_fla_res = [y for x in model_tag_res for y in x]
model_tag_fla_con = [y for x in model_con_res for y in x]

df["Tag"] = model_tag_fla_res
df["Confidence"] = model_tag_fla_con
df["2nd_Tag"] = model_2nd_tag_res
df["2nd_confidence"] = model_2nd_confidence

df.to_csv(r'Tag_results.csv', index=None, header=True)  # Don't forget to add '.csv' at the end of the path

def load_model_tag(x):
    interpreter = Interpreter.load('models/current/nlu_model')
    x = interpreter.parse(x)
    print(x)

load_model_tag("")


# ! python -m spacy download en
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()
doc = nlp(
    "Hi,     I got a Samung TV series 7 with HDR support, a Vizio soundar connected by HDMI with Atmos support and an apple TV 4K connected to the sound bar by HDMI.     My issue is that I can have HDR or Atmos, but no both at the same time. When I launch netflix from the apple TV and HDR is turned on, the movie starts flickering.     If I launch the built-in Netflix app I can get HDR but no Atmos.     I already talked to my set-box support and the soundbar manufacturer as well, but no one gives me a solution.     I tried with different ports, different cables, connecting directly the apple TV to my TV, resetting to factory settings, but nothing works.     TV specs:Model UE50NU7020Soft version: 1252     This is my last try, could be something related to the TV? Any setting?     Thanks  ")
for ent in doc.ents:
    print(ent.text, ent.label_)
    # doc.similarity(nlp("request"))


from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

config = RasaNLUConfig(cmdline_args={"pipeline": "spacy_sklearn"})
trainer = Trainer(config)
interpreter = trainer.train(training_data)

