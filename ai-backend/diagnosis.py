
####### MAIN CODE BOTTOM ########


from flask import Flask, send_from_directory, current_app, send_file, request, jsonify
from flask_cors import CORS

#################### All imports ###################
import numpy as np
import pandas as pd
import csv
import json
import pickle
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import itertools
from itertools import chain
import re

from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from googletrans import Translator

import openai as oa

import warnings

warnings.filterwarnings("ignore")

# -----------------------


##################### DATASET IMPORTS ###########################
def load_dataset(path):
    """
    path: Path to dataset file to be loaded
    """
    return pd.read_csv(path)


def check_dataset(data):
    """
    data: Data to be checked
    Prints part of the dataset to assert its correctness
    """
    print(data.head())


train_df = load_dataset("Data/train_final.csv")
test_df = load_dataset("Data/test_final.csv")

# Create a disease - symptoms pair from the training dataset
disease_list = []
symptoms = []  # this contains a list of lists

for i in range(len(train_df)):
    symptoms.append(train_df.columns[train_df.iloc[i] == 1].to_list())
    disease_list.append(train_df.iloc[i, -1])

# get all symptoms columns. This is the set of all unique symptoms
symptom_cols = list(train_df.columns[:-1])


# a helper function to preprocess the symptoms: remove underscores, etc
def clean_symptom(symp):
    """
    symp: Symptom to clean
    Removes underscores, fullstops, etc
    """
    return (
        symp.replace("_", " ")
        .replace(".1", "")
        .replace("(typhos)", "")
        .replace("yellowish", "yellow")
        .replace("yellowing", "yellow")
    )


# Apply the clean_symptom method to all symptoms
all_symptoms = [clean_symptom(symp) for symp in (symptom_cols)]

######################## TEXT PREPROCESSING  ###############################
nlp = spacy.load("en_core_web_sm")


def preprocess(document):
    nlp_document = nlp(document)
    d = []
    for token in nlp_document:
        if not token.text.lower() in STOP_WORDS and token.text.isalpha():
            d.append(token.lemma_.lower())
    return " ".join(d)


# apply preprocessing to all the symptoms
all_symptoms_preprocessed = [preprocess(symp) for symp in all_symptoms]
# associates each preprocessed symp with the name of its original column
cols_dict = dict(zip(all_symptoms_preprocessed, symptom_cols))
########################### TRANSLATIONS ###############################
translator = Translator()


def translate(text, src_lan, dest_lang="en"):
    """
    text: Text to translate
    src_lan: Source language
    dest_lan: Destination language
    """
    ar = translator.translate(text, src=src_lan, dest=dest_lang).text
    return ar


############################ SYNTACTIC SIMILARITY #######################
# a helper function to calculate the Jaccard similarity
def jaccard_similary(string1, string2):
    list1 = string1.split(" ")
    list2 = string2.split(" ")
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# a helper function to calculate the syntactic similarity between the symptom and corpus
def syntactic_similarity(symptom, corpus):
    most_sim = []
    poss_sym = []
    for symp in corpus:
        s = jaccard_similary(symptom, symp)
        most_sim.append(s)
    ordered = np.argsort(most_sim)[::-1].tolist()
    for i in ordered:
        if does_exist(symptom):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None


# Returns all the subsets of this set. This is a generator.
def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item


# Sort list based on length
def sort(a):
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a


# find all permutations of a list
def permutations(s):
    permutations = list(itertools.permutations(s))
    return [" ".join(permutation) for permutation in permutations]


def does_exist(txt):
    txt = txt.split(" ")
    combinations = [x for x in powerset(txt)]
    a = sort(combinations)
    for comb in combinations:
        for sym in permutations(comb):
            if sym in all_symptoms_preprocessed:
                return sym
    return False


# a helper function to help determine list of symptoms that contain a given pattern
def check_pattern(enquired_pat, symp_list):
    pred_list = []
    ptr = 0
    patt = "^" + enquired_pat + "$"
    regexp = re.compile(enquired_pat)
    for item in symp_list:
        if regexp.search(item):
            pred_list.append(item)
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return ptr, None


#################################### SEMANTIC SIMILARITY ###########################
def word_sense_disambiguation(word, context):
    """Determines the meaning of a word based on the context it is used in"""
    sense = lesk(context, word)  # lesk is a WSD tool from the NLTK
    return sense


def semantic_distance(doc1, doc2):
    doc1_p = preprocess(doc1).split(" ")
    doc2_p = preprocess(doc2).split(" ")
    score = 0
    for token1 in doc1_p:
        for token2 in doc2_p:
            syn1 = word_sense_disambiguation(token1, doc1)
            syn2 = word_sense_disambiguation(token2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                if x is not None and (x > 0.1):
                    score += x
    return score / (len(doc1_p) * len(doc2_p))


def semantic_similarity(symptom, corpus):
    max_sim = 0
    most_sim = None
    for symp in corpus:
        d = semantic_distance(symptom, symp)
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim


# def my_simp(symptom, corpus):
#     max_sim = 0
#     nlp = spacy.load("en_core_web_lg")
#     doc = nlp(str(symptom))
#     most_sim = None
#     for sym in corpus:
#         d = doc.similarity(nlp(sym))
#         if d > max_sim:
#             most_sim=sym
#             max_sim = d
#     return max_sim, most_sim


def suggest_symptom(sympt):
    """Takes an expression from the user and suggests the possible symptom the user is referring to"""
    symp = []
    synonyms = wn.synsets(sympt)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symptoms_preprocessed)
        if res != 0:
            symp.append(sym1)
    print(list(set(symp)))
    return list(set(symp))


def one_hot_vector(client_sympts, all_sympts, s_cols):
    """receives client_symptoms and returns a dataframe with 1 for associated symptoms
    cleint_sympts: symptoms identified by user
    all_sympts: all symptoms in the dataset
    s_cols: feature names
    """
    df = np.zeros([1, len(all_sympts)])
    for sym in client_sympts:
        df[0, all_sympts.index(sym)] = 1
    return pd.DataFrame(df, columns=s_cols)


def contains(small, big):
    """Check to see if a small set is contained in a bigger set"""
    status = True
    for i in small:
        if i not in big:
            status = False
    return status


def sympts_of_disease(df, disease):
    """receives an illness and returns all symptoms"""
    tempt_df = df[df.prognosis == disease]
    m2 = (tempt_df == 1).any()
    return m2.index[m2].tolist()


def possible_diseases(l, diseases):
    poss_dis = []
    for dis in set(diseases):
        if contains(l, sympts_of_disease(train_df, dis)):
            poss_dis.append(dis)
    print("posssss", poss_dis)
    return poss_dis


################################ MODEL TRAINING #############################
X_train = train_df.iloc[:, :-1]
X_test = test_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
y_test = test_df.iloc[:, -1]


def create_model():
    KNN_model = KNeighborsClassifier()
    return KNN_model


def train(X_train, y_train):
    model = create_model()
    model.fit(X_train, y_train)
    # save model
    file = open("model/KNN.pickle", "wb")
    pickle.dump(model, file)
    file.close()


def load_model():
    file = open("model/KNN.pickle", "rb")
    model = pickle.load(file)
    return model


########################### DRIVER PROGRAM ######################
def get_user_info():
    """Get user credentials for authentication
    This will be replaced by a proper authentication method when we implement the interface
    """
    print("Please enter your Name \t\t", end=" >> ")
    username = input("")
    print("Hi ", username + " ...")
    return str(username)


def related_symptom(client_symp, lan):
    """Determines which of the symptoms the user is trying to express"""
    if len(client_symp) == 1:
        return client_symp[0]
    sen = translate("Searches related to input: ", "en", lan)
    for num, s in enumerate(client_symp):
        s = translate(clean_symptom(s), "en", lan)
        print(num, ": ", s)
    if num != 0:
        sen = translate(f"Select the one you meant (0 - {num}):  ", "en", lan)
        print(sen, end="")
        conf_input = int(input(""))
    else:
        conf_input = 0
    disease_input = client_symp[conf_input]
    return disease_input


def select_lang():
    """
    Allows user to use the program in three different languages:
    1. English
    2. Turkish
    3. Spanish
    This list can be expanded by simply adding more options
    """
    translator = Translator()
    train(X_train, y_train)
    lan = ""
    print("Please a language below to continue... ")
    selected_lan = ""
    while True:
        print("1. English (EN)\n2. Türkçe (TR)\n3. Espanol (ES)")
        try:
            selected_lan = int(input(""))
            if selected_lan == 1:
                lan = "en"
                break
            elif selected_lan == 2:
                lan = "tr"
                break
            elif selected_lan == 3:
                lan = "es"
                break
            else:
                print("Input must be an integer between 1 and 3")
        except:
            print("Please make a valid selection (1 - 3)")

    return lan


def main_program(username, all_sympts_col, lan):
    """Takes at least two symptoms from the user and tries to guess the remaining symptoms
    Default language is English
    """
    """
        Get the 1st symptom -> process it (feed it to NLP) -> check_pattern -> get the appropriate corresponding symptom 
        (if pattern check == 1 == syntactic similarity: symptom found)
    """

    sentence = "What symptom are you experiencing"
    sen = translate(sentence, "en", lan)
    print(f"{sen}, {username}?\t\t", end=" >>  ")
    sympt1 = input("")
    sympt1 = translate(sympt1, lan, "en")
    sympt1 = preprocess(sympt1)
    sim_1, psym1 = syntactic_similarity(sympt1, all_symptoms_preprocessed)
    if sim_1 == 1:
        psym1 = related_symptom(psym1, lan)

    """
        Get the 2nd syp ->> process it ->> check_pattern ->>> get the appropriate one 
        (if pattern check == 1 == syntactic similarity: symptom found)
    """

    sentence = "Enter any other symptom you have"
    sen = translate(sentence, "en", lan)
    print(f"{sen}, {username}\t\t", end=" >>  ")
    sympt2 = input("")
    sympt2 = translate(sympt2, lan, "en")
    sympt2 = preprocess(sympt2)
    sim_2, psym2 = syntactic_similarity(sympt2, all_symptoms_preprocessed)
    if sim_2 == 1:
        psym2 = related_symptom(psym2, lan)

    # if pattern check for syntactic similarity fails (pattern_check == 0 for sympt1 or sympt2)
    # then try semantic similarity

    if sim_1 == 0 or sim_2 == 0:
        sim_1, psym1 = semantic_similarity(sympt1, all_symptoms_preprocessed)
        sim_2, psym2 = semantic_similarity(sympt2, all_symptoms_preprocessed)

        # if semantic similarity for  sympt1 == 0 (no symptom found)
        # then suggest possible data symptoms based on all data and input symptom synonyms
        if sim_1 == 0:
            suggested = suggest_symptom(sympt1)
            quest = "Do you exprecience any"
            quest = translate(quest, "en", lan)
            print(quest)
            for res in suggested:
                print(translate(res, "en", lan) + "?")
                inp = input("")
                if translate(inp.lower(), lan, "en") == "yes":
                    psym1 = res
                    sim_1 = 1
                    break

        # if semantic similarity for sympt2 == 0 (no symptom found)
        # suggest possible data symptoms based on all data and input symptom synonyms
        if sim_2 == 0:
            suggested = suggest_symptom(sympt2)
            for res in suggested:
                sen = translate(f"Do you feel {res} ? >> (yes or no)", "en", lan)
                print(sen)
                inp = input("")
                if translate(inp.lower(), lan, "en") == "yes":
                    psym2 = res
                    sim_2 = 1
                    break

        # if no syntaxic semantic and suggestion found, return None and ask for clarification

        if sim_1 == 0 and sim_2 == 0:
            return None, None
        else:
            # if at least one symptom found ->> duplicate it and proceed
            if sim_1 == 0:
                psym1 = psym2
            if sim_2 == 0:
                psym2 = psym1
    # create patient symptom list
    all_sympts = [cols_dict[psym1], cols_dict[psym2]]

    # predict possible diseases
    diseases = possible_diseases(all_sympts, disease_list)
    status = False
    quest = "Are you experiencing any"
    quest = translate(quest, "en", lan)
    print(quest)
    print(diseases)
    for dis in diseases:
        if status == False:
            for sym in set(sympts_of_disease(train_df, dis)):
                if sym not in all_sympts:
                    s = clean_symptom(sym)
                    s = translate(s, "en", lan)
                    print(s + "?")
                    while True:
                        inp = input("")
                        res = translate(inp, lan, "en").lower().strip()
                        if res in ["yeah", "yes", "no"]:
                            break
                        else:
                            print(
                                translate(
                                    "Please provide a (yes/no) answer:", "en", lan
                                ),
                                end=" >> ",
                            )
                    res = translate(inp, lan, "en").lower().strip()
                    if res in ["yeah", "yes"]:
                        all_sympts.append(sym)
                        diseases = possible_diseases(all_sympts, disease_list)
                        print(diseases)
                        if len(diseases) == 1:
                            status = True
    KNN_model = load_model()
    model_sym = one_hot_vector(all_sympts, all_sympts_col, symptom_cols)
    model_sym.to_csv("predicturesym.csv")
    predicted_disease = KNN_model.predict(
        one_hot_vector(all_sympts, all_sympts_col, symptom_cols)
    )
    for i, s in enumerate(all_sympts):
        all_sympts[i] = translate(clean_symptom(s), "en", lan)
    msg = translate("According to the following symptoms", "en", lan)
    print(f"{msg} {all_sympts}")
    msg = translate(f"You may have {predicted_disease[0]}", "en", lan)
    print(msg)


def main():
    username = ""
    while len(username) < 1:
        username = input(
            "Please provide your username to continue >> ",
        )
    lan = select_lang()
    status = True
    while status:
        main_program(username, symptom_cols, lan)
        print("Do you wish to check for another process? (yes or no)", end=" >> ")
        inp = input("")
        if inp == "yes":
            continue
        else:
            print("Thanks for choosing us for your diagnosis...")
            print("Wishing you a speedy recover")
            break


# if __name__ == '__main__':
#    main()
#     #train(X_train, y_train)

usernames = ""
languagees = ""

app = Flask(__name__)
CORS(app)


@app.route("/getuser", methods=["POST"])
def getuser():
    data = request.get_json()
    usernames = data["username"]
    languagees = data["lang"]

    sentence = "What symptoms are you experiencing"
    sen = translate(usernames + ", " + sentence + "?", "en", languagees)

    return jsonify(sen)


@app.route("/getsympfirst", methods=["POST"])
def getsympfirst():
    data = request.get_json()
    symptom = data["symptom"]
    languagees = data["lang"]
    usernames = data["username"]
    symptom = translate(symptom, languagees, "en")
    symptom = preprocess(symptom)
    sym_1, psym_1 = syntactic_similarity(symptom, all_symptoms_preprocessed)
    if sym_1 == 1:
        psym_1 = related_symptom(psym_1, languagees)
    sentenceTwo = "Enter any other symptom you have"
    senTwo = translate(usernames + ", " + sentenceTwo, "en", languagees)
    return jsonify(sym_1, psym_1, senTwo)


@app.route("/getsympsecond", methods=["POST"])
def getsympsecond():
    data = request.get_json()
    symptom2 = data["symptom2"]
    languagees = data["lang"]
    symptom2 = translate(symptom2, languagees, "en")
    symptom2 = preprocess(symptom2)
    sim_2, psym2 = syntactic_similarity(symptom2, all_symptoms_preprocessed)
    if sim_2 == 1:
        psym2 = related_symptom(psym2, languagees)

    return jsonify(sim_2, psym2)


@app.route("/setsympsall", methods=["POST"])
def setsympsall():
    data = request.get_json()
    ps1 = data["psym_1"]
    ps2 = data["psym_2"]
    languagees = data["lang"]
    all_symptoms = [cols_dict[ps1], cols_dict[ps2]]
    diseasess = possible_diseases(all_symptoms, disease_list)
    question = "Are you experiencing any of these symptoms ?"
    question = translate(question, "en", languagees)
    addQuestion = "Do you have any other symptoms to add?"
    addQuestion = translate(addQuestion, "en", languagees)
    sym = list(set(sympts_of_disease(train_df, diseasess[0])))
    symptomListt = []
    for i in sym:
        new_string = i.replace("_", " ")
        symp = translate(new_string, "en", languagees)
        symptomListt.append(symp)

    aa = list(symptomListt)
    sPs1 = ps1.replace("_", " ")
    transPs1 = translate(sPs1, "en", languagees)
    sPs2 = ps2.replace("_", " ")
    transPs2 = translate(sPs2, "en", languagees)
    transSymptoms = [transPs1, transPs2]
    return jsonify(transSymptoms, diseasess, question, aa, addQuestion)


oa.api_key = "*****************************************************"


@app.route("/finaldiagnose", methods=["POST"])
def finaldiagnose():
    data = request.get_json()
    languagees = data["lang"]
    allSymptoms = data["allSymptoms"]
    asymp = []
    for i in allSymptoms:
        symp = translate(i, languagees, "en")
        asymp.append(symp)
    response = oa.Completion.create(
        # model name used here is text-davinci-003
        # there are many other models available under the
        # umbrella of GPT-3
        model="text-davinci-003",
        # passing the user input
        prompt="""My symptoms are {asymp[0]},{asymp[1]},{asymp[2]},{asymp[3]},{asymp[4]},{asymp[5]},{asymp[6]},{asymp[7]},{asymp[8]},{asymp[9]},{asymp[10]},{asymp[11]}. According to these symptoms what is diagnosis scenario""",
        # generated output can have "max_tokens" number of tokens
        max_tokens=100,
        # number of outputs generated in one call
        n=3,
    )
    output = response.choices[0].text.strip()
    output = translate(output, "en", languagees)
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
