import spacy

nlp = spacy.load("en_core_web_sm")

# text = "turn off light1"
text = input("Enter the Chat : ")
doc = nlp(text)
print("Sentence : " + doc.text + "\n")
print("Processed Details :")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.is_stop)

verbs = []
adj = []
noun = []
intent = ""
for token in doc:
    if (token.pos_ == 'VERB'):
        verbs.append(token)
    if (token.pos_ == 'ADJ'):
        adj.append(token)
    if (token.pos_ == 'NOUN'):
        noun.append(token)

print("Verbs : " + str(verbs))
print("Noun : " + str(noun))
for i in range(len(verbs)):
    intent = intent + str(verbs[i])

for j in range(len(noun)):
    intent = intent + str(noun[j])

print()
print("Intent : " + intent)
