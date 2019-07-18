# -*- coding: utf-8 -*-
"""
CLUST -TXT XXS AZURE
"""

import sys
import logging
import time
import datetime
import os
import re
from random import randint
import math
import collections

# basic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# nltk
# import nltk
# from nltk.corpus import stopwords
# from nltk import word_tokenize, sent_tokenize

# spacy
import spacy
import en_core_web_sm

# gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import multiprocessing

# sklearn
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# clustering
from pylab import plot, show

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from scipy.cluster.vq import kmeans, vq

# %% set up logging and time management module

"""
###################################################################
                    TIME AND LOG MANAGEMENT
###################################################################
"""


def take_time(start_time=time.time(), end_time=time.time(), process='Unknown'):
    elapsed_time = end_time - start_time
    total_time = str(datetime.timedelta(seconds=elapsed_time))
    # df_time.append({ 'process':process, 'start_time':start_time, 'end_time':end_time, 'total_time':total_time}, ignore_index=True)
    print("Process %", process)
    print("START %s" % start_time)
    print("END %s" % end_time)
    print("TOTAL %s" % total_time)
    return {'process': process, 'total_time': total_time}


# %% user defined parameters

"""
###################################################################
                   SET USER DEFINED PARAMETERS                    
###################################################################
"""

def FileForPrinting(path):
    """
    with open(path, "w+", encoding='utf-8') as text_file:
        print(f'{str(len(sys.argv))} /n', file=text_file)
        for i_index in range(0, len(sys.argv) - 1):
            print(f'{str(sys.argv[i_index]} /n', file=text_file)
    """
    with open(path, "w+", encoding='utf-8') as text_file:
        print(f'{str(11111)} /n', file=text_file)
        for i_index in range(0, len(sys.argv) - 1):
            print(f'{str(sys.argv[i_index]} /n', file=text_file)

def SetParameters(df_time):
    print(__doc__)
    print("SCRIPT START")
    arg_names = ['file_path', 'experiment', 'experiment_ref', 'parameter_type', 'model', 'dim_reduction', 'pipeline']
    
    FileForPrinting(os.path.join(os.getcwd() + '/ConsoleArgs.txt'))
    
    args = dict(zip(arg_names, sys.argv))
    default = False
    # print("Python File:", args['file_path'])

    if (len(args) <= 1 or len(args) < len(arg_names)):
        sys.argv.append(20)      # experiment number
        sys.argv.append(0)      # experiment_ref
        sys.argv.append(0)      # parameter_type
        sys.argv.append(2)      # model
        sys.argv.append(1)      # dim_reduction
        sys.argv.append("5")    # pipeline flow
        default = True
        args = dict(zip(arg_names, sys.argv))  # se guardan en un dictionary
    else:
        print("Reading Arguments")

        for i, key_value in enumerate(args.items()):
            value = key_value[1]
            if (key_value[1] == "-1" and i > 0):
                key = key_value[0]
                args[key] = 0
                print(key)
                print(value)

    # current working directory
    if (args['file_path'] != None):
        file_path = args['file_path']
        base_path = os.path.join(os.path.dirname(args['file_path']) + '/')
    else:
        base_path = os.path.join(os.getcwd() + '/')  # gets current working directory

    docs_name = ['AM.xlsx', 'english_tickets_1.xlsx', 'english_tickets_2.xlsx', 'spanish_tickets.xlsx',
                 'descripcion.xlsx', 'alstrom_descriptions.xlsx', 'historical_incidents.xlsx']
    doc = ""
    # experiment_ref=int(args['experiment_ref'])
    if 'doc' in args:
        doc_name = docs_name[args['doc']]
    else:
        if not default:
            doc_name = 'EXPERIMENT_' + str(args['experiment'])
            doc = args['experiment']
        else:
            doc_name = docs_name[0]

    print(doc_name)

    """
    model to choose
    0- faster
    1- mix
    2- paper
    3- paper-tuned
    """

    model = int(args['model'])  # 0-new , 1-pretrained, 2- own_pretrained_model

    # %% adjust parameters
    """
    *****************
      change to default arguments if needed
    ****************
    """
    # en default model es 0
    if (model == 0 or (model != 1 and model != 2) or model == -1):
        pretrained_emb = False
        own_model = False
        train = True
        model = "new"
    else:
        if (model == 1):
            pretrained_emb = True
            own_model = False
            train = False
            model = 'pretrained_emb'
        else:
            if (model == 2):
                pretrained_emb = False
                own_model = True
                train = False
                model = 'own_model'

    # %% print in console and set EXPERIMENT
    # print("Parameters sent DATAFRAME")
    # print("base_path ", base_path)
    # paper trained default

    print("Parameters sent")
    print("experiment: ", args['experiment'])
    print("experiment_ref: ", args['experiment_ref'])
    print("parameter_type: ", int(args['parameter_type']))  # faster 0, middle 1, accuracy- 2, acccuracy-tuned 3
    print("dimensionality reduction ", int(args['dim_reduction']))  # 1-pca , 2-tsne
    print("pipeline: ", args['pipeline'])
    print("model: ", model)

    experiment_stage = "EXPERIMENT_"
    experiment_name = experiment_stage + str(args['experiment'])
    print('Experiment name: ' + experiment_name)

    end_time = time.time()
    df_time = df_time.append(take_time(start_time, end_time, 'SET PARAMETERS'), ignore_index=True)
    print(df_time)

    # %% storage
    """
    ###################################################################
                                STORAGE
    ###################################################################
    """

    """
    ****************************
       MAIN FOLDERS
    ****************************
    """

    if not os.path.exists(os.path.join(base_path + "Datasets")):
        os.makedirs(os.path.join(base_path + "Datasets"))

    if not os.path.exists(os.path.join(base_path + "Models")):
        os.makedirs(os.path.join(base_path + "Models"))

    if not os.path.exists(os.path.join(base_path + '/' + "Metadata")):
        os.makedirs(os.path.join(base_path + '/' + "Metadata"))

    if not os.path.exists(os.path.join(base_path + "Results")):
        os.makedirs(os.path.join(base_path + "Results"))

    if not os.path.exists(os.path.join(base_path + '/' + "Clustering")):
        os.makedirs(os.path.join(base_path + '/' + "Clustering"))

    if not os.path.exists(os.path.join(base_path + '/ ' + "Dim Reduction")):
        os.makedirs(os.path.join(base_path + '/ ' + "Dim Reduction"))

    """
    *************
    EXPERIMENT SPECIFIC PATHS
    *************
    """
    # in results folder
    if not os.path.exists(os.path.join(base_path + "Datasets" + '/' + "EXPERIMENT_" + str(args['experiment']))):
        os.makedirs(os.path.join(base_path + "Datasets" + '/' + "EXPERIMENT_" + str(args['experiment'])))

    if not os.path.exists(os.path.join(base_path + "Datasets" + '/' + "PREPROCESS_" + experiment_name)):
        os.makedirs(os.path.join(base_path + "Datasets" + '/' + "PREPROCESS_" + experiment_name))

    if not os.path.exists(os.path.join(base_path + '/' + "Metadata" + '/' + "METADATA_" + experiment_name)):
        os.makedirs(os.path.join(base_path + '/' + "Metadata" + '/' + "METADATA_" + experiment_name))

    if not os.path.exists(os.path.join(base_path + '/' + "Clustering" + '/' + "CLUSTERING_" + experiment_name)):
        os.makedirs(os.path.join(base_path + '/' + "Clustering" + '/' + "CLUSTERING_" + experiment_name))

    if not os.path.exists(os.path.join(base_path + '/ ' + "Dim Reduction" + '/' + "DIM_REDUCTION_" + experiment_name)):
        os.makedirs(os.path.join(base_path + '/ ' + "Dim Reduction" + '/' + "DIM_REDUCTION_" + experiment_name))
    
    if not os.path.exists(os.path.join(base_path + '/ ' + "Results" + '/' + experiment_name)):
        os.makedirs(os.path.join(base_path + '/ ' + "Results" + '/' + experiment_name))

    """
    *************
    SAVE IN RESULTS_EXPERIMENT FILES
    *************
    """
    if (default):
        file_path = os.path.join(base_path + "Datasets" + '/' + doc_name)
    else:
        file_path = os.path.join(base_path + "Datasets" + '/' + doc_name + '.xlsx')

    """
    #global path 
    model_file_path = os.path.join(base_path + "Models" + '/' + "Model_" + str(experiment_name))
    model_doctag_file_path = os.path.join(base_path + "Models" + '/' + "model_doctag")

    #results
    training_data_path = os.path.join(base_path + "Results" + '/' + "EXPERIMENT_" + str(args['experiment']) + '/' + "training_data.csv")
    test_data_path = os.path.join(base_path + "Results" + '/' + "EXPERIMENT_" + str(args['experiment']) + '/' + "test_data.txt")
    similar_path = os.path.join(base_path + "Results" + '/' + "EXPERIMENT_" + str(args['experiment']) + '/' + "similar_random.csv")

    #metadata
    parameters_file_path = os.path.join(base_path + '/' + "Metadata" + '/' + "METADATA_" + experiment_name + '/' + experiment_name + "_parameters")

    #clustering 
    clustering_pca_file_path = os.path.join(base_path + '/' + "Clustering" + '/' + "CLUSTERING_" + experiment_name + '/' + "clustering_PCA.txt")
    clustering_tsne_file_path = os.path.join(base_path + '/' + "Clustering" + '/' + "CLUSTERING_" + experiment_name + '/' + "clustering_TSNE.txt")
    """

    resp = {
        "doc": doc,
        "doc_name": doc_name,
        "dim_reduction": int(args['dim_reduction']),
        "pretrained_emb": pretrained_emb,
        "own_model": own_model,
        "train": train,
        "model": model,
        "base_path": base_path,
        "experiment_name": experiment_name,
        "args": args,
        "file_path": file_path,
        "df_time": df_time
    }

    return resp


# %% MAIN
"""
###################################################################
                        PREPROCESSING
###################################################################
"""
"""
    Preprocessing techniques:
       - Tokenization 
       - Stop Word Removal
       - Lower Case and Punctuation or Number removal 
"""


def Preprocessing(df_time, base_path, experiment_name, args, file_path, pipe_flow):
    # loading model
    nlp = spacy.load('en')
    # fetch data
    start_time = time.time()  # contamos el tiempo
    file = FetchData(file_path)  # metodo para sacar los datos del archivo
    end_time = time.time()  # terminamos el conteo del tiempo
    df_time = df_time.append(take_time(start_time, end_time, 'FILE READING 1'), ignore_index=True)

    print('File shape ', file.shape)

    vocab_size = file.shape[0] - 1
    file = shuffle(file)
    train = False

    start_time = time.time()
    """
    Se va a seguir el flujo empleado por lo que el usuario escoge
    Default es mi preferencia
    """
    
    for pipe in pipe_flow:
        if pipe != ' ':
            if pipe == "1":
                print('Pipeline One')
                pipeline = CleaningCorpora(nlp, file, vocab_size)           #stop words, -
            if pipe == "2":
                print('Pipeline Two')
                pipeline = CleaningCorporaTwo(nlp, file, vocab_size)
            if pipe == "3":
                print('Pipeline Three')
                pipeline = CleaningCorporaThree(nlp, file, vocab_size)     #greetings y farewells
            if pipe == "4":
                print('Pipeline Four')
                pipeline = CleaningCorporaFour(nlp, file, vocab_size)
            if pipe == "5":
                print('Pipeline Five')
                pipeline = CleaningCorporaThree(nlp, file, vocab_size, base_path)
                pipeline = CleaningCorpora(nlp, pd.DataFrame({'col':pipeline['corpus_preprocessed']}), vocab_size)
                pipeline = CleaningCorporaFour(nlp, pd.DataFrame({'col':pipeline['corpus_preprocessed']}), vocab_size)
    
    #preprocessing = CleaningCorpora(nlp, file, vocab_size)  # stopwords and NER
    #pipeplines
    #pipelineOne = CleaningCorpora(nlp, file, vocab_size)           #stop words, -
    #pipelineTwo = CleaningCorporaTwo(nlp, file, vocab_size)
    #pipelineThree = CleaningCorporaThree(nlp, file, vocab_size)     #greetings y farewells
    #pipelineFour = CleaningCorporaFour(nlp, file, vocab_size)
    # corpus_preprocessed    corpus_raw    entities
    
    print("Preprocessing Finished")
    # split data
    # df_file['training_data'] = df_file[:math.floor(vocab_size * .8)]
    # df_file['test_data'] = df_file[math.ceil(vocab_size * .2):]

    docs_used = pd.DataFrame({'preprocessed_sentence': pipeline['corpus_preprocessed']})
    model = pipeline['corpus_preprocessed']
    model = pd.DataFrame({'preprocessed_sentence': model[:int(vocab_size * 0.8)]})
    train = pipeline['corpus_preprocessed']
    train = pd.DataFrame({'preprocessed_sentence': train[int(vocab_size * 0.8):]})
    docs_original = pd.DataFrame({'sentences': pipeline['corpus_raw']})
    docs_xxs = pd.DataFrame({'entities': pipeline['entities']})

    docs_used.to_csv(os.path.join(base_path + "Datasets" + '/' + "PREPROCESS_" + experiment_name + '/' + "dataset.txt"),
                     sep='\t', encoding='utf-8', index=False, header=None)
    model.to_csv(os.path.join(base_path + "Datasets" + '/' + "PREPROCESS_" + experiment_name + '/' + "modelSample.txt"),
                     sep='\t', encoding='utf-8', index=False, header=None)
    train.to_csv(os.path.join(base_path + "Datasets" + '/' + "PREPROCESS_" + experiment_name + '/' + "trainSample.txt"),
                     sep='\t', encoding='utf-8', index=False, header=None)
    docs_xxs.to_csv(os.path.join(base_path + "Datasets" + '/' + "PREPROCESS_" + experiment_name + '/' + "sentences_xxs.txt"),
                    encoding='utf-8', index=False, header=None)
    """
    docs_original.to_csv(os.path.join(base_path + "Results" + '/' + experiment_name + '/' + "original_dataset.csv"),
                         encoding='utf-8', index=False, header=None)
    """
    end_time = time.time()
    df_time = df_time.append(take_time(start_time, end_time, 'PREPROCESSING '), ignore_index=True)

    end_time = time.time()
    df_time = df_time.append(take_time(start_time, end_time, 'NER + StopWords'), ignore_index=True)
    # regresamos el vocab_size, training y resp
    data = {
        "vocab_size": vocab_size,
        "train": train,
        "pipeline": pipeline,
        "df_time": df_time
    }

    return data


def FetchData(file_path):
    print("Fetching data from file in ", file_path)
    df_file = pd.read_excel(file_path, usecols="A")

    return df_file

def FetchInfo(file_path):
    print("Fetching data from file in ", file_path)
    txt_file = open(file_path, 'r')
    
    return txt_file

#vamos a eliminar los greetings
def RemoveGreetings(sentence, greetings):
    try:
        #checamos si tiene formato de espacios
        if '\n' in sentence:
            #hacemos el split
            greeting = sentence.split('\n')
            for i in range(len(greeting) - 1):
                words = greeting[i].split(' ')
                if words[0].lower() in greetings:
                    #tenemos en el dictionary la primera palabra
                    #es un greeting
                    #checamos si tiene una coma
                    if ',' in greeting[i]:
                        #eliminamos desde el greeting hasta la coma
                        index = greeting[i].find(',')
                        #sentence = sentence.replace('\n', ' ')
                        substring = greeting[i]
                        #print(substring[0:index+1])
                        sentence = sentence.replace(substring[0:index+1], '')
                        sentence = RemoveWhiteSpaces(sentence)
                    else:
                        #no tiene coma, solo eliminamos el greeting encontrado
                        #checamos el length del greeting
                        if len(greeting[i]) >= len(sentence) / 4:
                            #eliminamos solo el greeting
                            #sentence = sentence.replace('\n', ' ')
                            word = greeting[i].split(' ')
                            index = greeting[i].lower().find(word[0].lower())
                            substring = greeting[i]
                            #sentence = sentence.replace(word[0], '')
                            sentence = sentence.replace(substring[index:len(word[0])], '')
                            sentence = RemoveWhiteSpaces(sentence)
                            #print(word[0])
                        else:
                            #se elimina todo
                            #sentence = sentence.replace('\n', ' ')
                            sentence = sentence.replace(greeting[i], '')
                            sentence = RemoveWhiteSpaces(sentence)
                            #print(greeting[0])
                    #encontró el greeting, nos salimos del ciclo
                    break
                else:
                    #not in greetings
                    #checamos si es parte del greeting
                    for key, value in greetings.items():
                        if words[0].lower() in key.lower():
                            #verificamos que tenga la mayor parte del contenido
                            if len(words[0]) >= len(key) / 2 and len(words[0]) != 1:
                                #checamos hasta la coma
                                if ',' in greeting[i]:
                                    #eliminamos desde el greeting hasta la coma
                                    index = greeting[i].find(',')
                                    #sentence = sentence.replace('\n', ' ')
                                    substring = greeting[i]
                                    #print(substring[0:index+1])
                                    sentence = sentence.replace(substring[0:index+1], '')
                                    sentence = RemoveWhiteSpaces(sentence)
                                else:
                                    #si todo es mayor que tres veces el greeting
                                    if len(greeting[i]) >= len(words[0]) * 3:
                                        #se elimina solo el greeting
                                        index = greeting[i].lower().find(words[0].lower())
                                        substring = greeting[i]
                                        #sentence = sentence.replace(words[0], '')
                                        sentence = sentence.replace(substring[index:len(words[0])], '')
                                        sentence = RemoveWhiteSpaces(sentence)
                                    else:
                                        #se elimina todo
                                        sentence = sentence.replace(greeting[i], '')
                                        sentence = RemoveWhiteSpaces(sentence)
                            #encontró el greeting, nos salimos del ciclo
                            break
                        else:
                            #todos los posibles greetings que tienen cambios de formato
                            if key.lower() in words[0].lower():
                                #checamos hasta la coma
                                if ',' in greeting[i]:
                                    #eliminamos desde el greeting hasta la coma
                                    index = greeting[i].find(',')
                                    #sentence = sentence.replace('\n', ' ')
                                    substring = greeting[i]
                                    #print(substring[0:index+1])
                                    sentence = sentence.replace(substring[0:index+1], '')
                                    sentence = RemoveWhiteSpaces(sentence)
                                else:
                                    #checamos que empiece con el greeting
                                    index = greeting[i].lower().find(key.lower())
                                    #Si empieza con el greeting
                                    if index == 0:
                                        #si es 2/3 de la palabra
                                        if len(key) >= len(words[0]) * 0.666:
                                            #eliminamos este greeting
                                            index = greeting[i].lower().find(words[0].lower())
                                            substring = greeting[i]
                                            #sentence = sentence.replace(words[0], '')
                                            sentence = sentence.replace(substring[index:len(words[0])], '')
                                            sentence = RemoveWhiteSpaces(sentence)
                                #encontró el greeting, nos salimos del ciclo
                                break
        return sentence
    except TypeError:
        print(sentence)
        return sentence

def RemoveFarewells(sentence, farewells):
    breakBool = False
    try:
        #checamos si tiene formato
        if '\n' in sentence:
            #hacemos el split
            farewell = sentence.split('\n')
            #pensar en una forma para poder checar el penúltimo también
            #muchas veces ponen Thank you y luego firman con su nombre
            for line in farewell:
                for f in farewells:
                    if f.lower() in line.lower():
                        #aqui nos detenemos y cortamos todo
                        #validamos la linea que sea
                        if line != farewell[0]:
                            #no es la primera linea
                            #checamos la posición del farewell
                            index = line.lower().find(f.lower())
                            if index >= len(line) / 2:
                                #el farewell está pasando la mitad de la linea
                                #cortamos del farewell al final
                                substring = sentence[sentence.lower().find(f.lower()):]
                                sentence = sentence.replace(substring, '')
                                sentence = RemoveWhiteSpaces(sentence)
                                breakBool = True
                                break
                            else:
                                #se elimina toda la linea y hasta el final
                                substring = sentence[sentence.lower().find(line.lower()):]
                                sentence = sentence.replace(substring, '')
                                sentence = RemoveWhiteSpaces(sentence)
                                breakBool = True
                                break
                if breakBool:
                    break
        return sentence
    except TypeError:
        return sentence

# vamos a eliminar cambios en formatos e imperfecciones
def RemoveImperfections(sentence):
    try:
        # checamos los cambios en el formato (remove tabs and enters)
        if '\t' in sentence:
            sentence = sentence.replace('\t', ' ')
        if '\n' in sentence:
            sentence = sentence.replace('\n', ' ')
        if '"' in sentence:
            sentence = sentence.replace('"', '')
        if "'" in sentence:
            sentence = sentence.replace("'", '')
        if '|' in sentence:
            sentence = sentence.replace('|', ' ')
        # checamos que tenga un espacio después de
        if '..' in sentence:
            sentence = sentence.replace('..', '.')
        if ',' in sentence:
            sentence = sentence.replace(',', ', ')
        if ' ,' in sentence:
            sentence = sentence.replace(' ,', ',')
        if ';' in sentence:
            sentence = sentence.replace(';', '; ')
        if '-->' in sentence:
            sentence = sentence.replace('-->', ' ')
        if '>' in sentence:
            sentence = sentence.replace('>', ' ')
        if '<' in sentence:
            sentence = sentence.replace('<', ' ')
        if '->' in sentence:
            sentence = sentence.replace('->', ' ')
        if ' + ' in sentence:
            sentence = sentence.replace(' + ', ' and ')
        if ' & ' in sentence:
            sentence = sentence.replace(' & ', ' and ')
        if ' - ' in sentence:
            sentence = sentence.replace(' - ', ' ')
        # if ':' in sentence:
        # sentence = sentence.replace(':', ': ')
        if ' :' in sentence:
            sentence = sentence.replace(' :', ': ')
        if ' : ' in sentence:
            sentence = sentence.replace(' : ', ': ')
        if ' .' in sentence:
            sentence = sentence.replace(' .', '.')
        if '(' in sentence:
            #sentence = sentence.replace('(', ' ( ')
            sentence = sentence.replace('(', '')
        if ')' in sentence:
            #sentence = sentence.replace(')', ' ) ')
            sentence = sentence.replace(')', '')
        if 'pls' in sentence:
            sentence = sentence.replace('pls', 'please')
        if 'Pls' in sentence:
            sentence = sentence.replace('Pls', 'Please')
        if 'pls.' in sentence:
            sentence = sentence.replace('pls.', 'please')
        if 'Pls.' in sentence:
            sentence = sentence.replace('Pls.', 'Please')
        # quitamos todo doble espacio que pudimos haber causado
        if "   " in sentence:
            sentence = sentence.replace("   ", " ")
        if "  " in sentence:
            sentence = sentence.replace("  ", " ")
            
        return sentence
    except TypeError:
        return sentence

def RemoveDoubleSpaces(sentence):
    if "   " in sentence:
        sentence = sentence.replace("   ", " ")
    if "  " in sentence:
        sentence = sentence.replace("  ", " ")
        
    return sentence


def RemoveWhiteSpaces(word):
    try:
        if word[0] == ' ':
            word = word[1:]
            return RemoveWhiteSpaces(word)
        else:
            if word[len(word) - 1] == ' ':
                word = word[:len(word) - 2]
                return RemoveWhiteSpaces(word)
            else:
                return word
    except IndexError:
        # print('Remove White Spaces catch', word, len(word))
        return word


def CleaningCorpora(nlp, file, vocab_size):
    nlp = en_core_web_sm.load()
    corpus_preprocessed = []  # texto preprocesado
    corpus_raw = []  # texto original
    entities = {}  # xxx potenciales
    hyphEntities = {}

    for index in range(0, vocab_size):
        corpus_raw.append(file.iat[index, 0])
        preprocessed = RemoveImperfections(corpus_raw[index])
        sentence = nlp(preprocessed)

        i = 0  # contadores auxiliares para encontrar parejas NNP
        hyphFlag = False
        # POS Tagger
        NPPwords = []
        for token in sentence:
            # se agrega uno a i aunque no sea NNP
            i = i + 1
            # checamos otro tipo de token
            if token.text == '-':
                hyphFlag = True
            else:
                text = token.text
                if text[len(text) - 1] == ':':
                    try:
                        # el token que sigue será la entity
                        wordToken = RemoveWhiteSpaces(sentence[i].text.lower())
                        NPPwords.append(wordToken)
                        # entities = SaveEntities(sentence[i + 1].text.lower(), entities)
                    except IndexError:
                        pass
            # token.tag_ == 'HYPH'
            if token.tag_ == 'NNP' and token.text != '-':
                # checamos que no sea el primero
                if i == len(sentence):
                    # checamso que no sea la del después de los :
                    if len(NPPwords) >= 1:
                        wordToken = RemoveWhiteSpaces(token.text.lower())
                        if NPPwords[len(NPPwords) - 1] == wordToken:
                            # lo eliminamos del arreglo, al cabo se vuelve a agregar
                            # esto es para que no se repita
                            NPPwords.pop()
                    # grabamos el NNP en el arreglo
                    NPPword = RemoveWhiteSpaces(token.text.lower())
                    NPPwords.append(NPPword)

                    # es la última, entonces se graba
                    NPPword = ''
                    if len(NPPwords) > 1:
                        NPPword = NPPwords[len(NPPwords) - 1] + ' '
                        NPPwords.pop()
                        for word in reversed(NPPwords):
                            NPPword += word + ' '
                    else:
                        NPPword = NPPwords[0]

                    NPPword = RemoveWhiteSpaces(NPPword)

                    if len(NPPword) > 0:
                        # mandamos como token los NPPs
                        entities = SaveEntities(NPPword, entities)

                    # limpiamos el arreglo
                    NPPwords = []
                else:
                    # es el primero
                    NPPword = RemoveWhiteSpaces(token.text.lower())
                    NPPwords.append(NPPword)
            else:
                # no es NPP
                # grabamos los NPPwords que llegamos
                if len(NPPwords) != 0:
                    NPPword = ''
                    if len(NPPwords) > 1:
                        NPPword = NPPwords[len(NPPwords) - 1] + ' '
                        NPPwords.pop()
                        for word in reversed(NPPwords):
                            NPPword += word + ' '
                    else:
                        NPPword = NPPwords[0]

                    NPPword = RemoveWhiteSpaces(NPPword)
                    if len(NPPword) > 0:
                        # mandamos como token los NPPs
                        entities = SaveEntities(NPPword, entities)
                    # limpiamos el arreglo
                    NPPwords = []
                # else no hay nada que grabar
            if token.is_stop:
                # la eliminamos del texto
                preprocessed = preprocessed.replace(' ' + token.text + ' ', ' ')
        # eliminamos posibles cambios de formato
        preprocessed = RemoveDoubleSpaces(preprocessed)
        # ya se termino de procesar la palabra
        # sacamos los posibles entities obtenidos de los -
        if hyphFlag:
            posEntities = HYPHEntity(preprocessed)

            for posEntity in posEntities:
                if posEntity in hyphEntities:
                    hyphEntities.update({posEntity: hyphEntities[posEntity] + 1})
                else:
                    hyphEntities.update({posEntity: 1})
                entities = SaveEntities(posEntity, entities)
        corpus_preprocessed.append(preprocessed)

    # filtramos el dictionary
    filterBy = [' ', '/', '\\', '.', '_', '-', '+', ',', ':']  # filtramos por varios requisitos

    for filterParam in filterBy:
        entities = FilterEntities(entities, filterParam)
    # ordenamos el diccionario
    # sorted(entities.values())

    # quiero ver los datos
    after = SeeData(entities)

    # vamos a eliminar todos los que tengan frecuencia 1
    # a excepcion de los alfanumericos o numericos
    entities = DeleteEntities(entities)

    before = SeeData(entities)

    # eliminamos todos los que son parte de las entidades de los '-'
    # checamos si existen solas o si alguna contiene otra
    repeatedEntities = {}
    for hyphEntity in hyphEntities:
        separatedEntity = hyphEntity.split('-')
        for entity in separatedEntity:
            if entity in entities:
                # si existe por si sola
                # hay que eliminarla
                del entities[entity]
            else:
                # checamos si lo contiene
                for key, value in entities.items():
                    if entity in key:
                        repeatedEntities.update({key: value})
                        # del entities[key]
                # for key in repeatedEntities:
                # del entities[key]

    data = {
        "corpus_preprocessed": corpus_preprocessed,
        "corpus_raw": corpus_raw,
        "entities": entities,
        "after": after,
        "before": before,
        "hyphEntities": hyphEntities,
        "repeatedEntities": repeatedEntities
    }

    return data


def CleaningCorporaTwo(nlp, file, vocab_size):
    nlp = en_core_web_sm.load()
    corpus_preprocessed = []  # texto preprocesado
    corpus_raw = []  # texto original
    entities = {}  # xxx potenciales
    hyphEntities = {}
    verbList = {}   #lista de verbos
    new_verbs = {}    #dic de verbos nuevos
    word_data = {}  #ver como quedo la palabra
    
    #lista de verbos
    verb_text = FetchInfo("C:/Users/carlos.ortega/Source/Repos/AutomationPlatform/TuringExpo/bin/Debug/Python/Datasets/verbs.txt")
    verb_list = verb_text.read().split('\n')
    verbs = {}
    for i in range(0, len(verb_list) - 1):
        verbs.update({ verb_list[i]: 0})

    for index in range(0, vocab_size):
        corpus_raw.append(file.iat[index, 0])
        preprocessed = RemoveImperfections(corpus_raw[index])
        sentence = nlp(preprocessed)

        i = 0  # contadores auxiliares para encontrar parejas NNP
        hyphFlag = False
        # POS Tagger
        NPPwords = []
        for token in sentence:
            # se agrega uno a i aunque no sea NNP
            i = i + 1
            # checamos otro tipo de token
            if token.text == '-':
                hyphFlag = True
            else:
                text = token.text
                if text[len(text) - 1] == ':':
                    try:
                        # el token que sigue será la entity
                        wordToken = RemoveWhiteSpaces(sentence[i].text.lower())
                        NPPwords.append(wordToken)
                        # entities = SaveEntities(sentence[i + 1].text.lower(), entities)
                    except IndexError:
                        pass
            # token.tag_ == 'HYPH'
            if token.tag_ == 'NNP' and token.text != '-':
                # checamos que no sea el primero
                if i == len(sentence):
                    # checamso que no sea la del después de los :
                    if len(NPPwords) >= 1:
                        wordToken = RemoveWhiteSpaces(token.text.lower())
                        if NPPwords[len(NPPwords) - 1] == wordToken:
                            # lo eliminamos del arreglo, al cabo se vuelve a agregar
                            # esto es para que no se repita
                            NPPwords.pop()
                    # grabamos el NNP en el arreglo
                    NPPword = RemoveWhiteSpaces(token.text.lower())
                    NPPwords.append(NPPword)

                    # es la última, entonces se graba
                    NPPword = ''
                    if len(NPPwords) > 1:
                        NPPword = NPPwords[len(NPPwords) - 1] + ' '
                        NPPwords.pop()
                        for word in reversed(NPPwords):
                            NPPword += word + ' '
                    else:
                        NPPword = NPPwords[0]

                    NPPword = RemoveWhiteSpaces(NPPword)

                    if len(NPPword) > 0:
                        # mandamos como token los NPPs
                        entities = SaveEntities(NPPword, entities)

                    # limpiamos el arreglo
                    NPPwords = []
                else:
                    # es el primero
                    NPPword = RemoveWhiteSpaces(token.text.lower())
                    NPPwords.append(NPPword)
            else:
                # no es NPP
                # grabamos los NPPwords que llegamos
                if len(NPPwords) != 0:
                    NPPword = ''
                    if len(NPPwords) > 1:
                        NPPword = NPPwords[len(NPPwords) - 1] + ' '
                        NPPwords.pop()
                        for word in reversed(NPPwords):
                            NPPword += word + ' '
                    else:
                        NPPword = NPPwords[0]

                    NPPword = RemoveWhiteSpaces(NPPword)
                    if len(NPPword) > 0:
                        # mandamos como token los NPPs
                        entities = SaveEntities(NPPword, entities)
                    # limpiamos el arreglo
                    NPPwords = []
                # else no hay nada que grabar
            if token.pos_ == 'VERB':
                if '/' in token.text or '-' in token.text or '\\' in token.text or '.' in token.text:
                    #vemos si podemos partir el token
                    aux = SplitData(token.text.lower(), verbs, nlp)
                    if 'word' in aux.keys():
                        preprocessed = preprocessed.replace(' ' + aux['word'] + ' ', ' ')
                        word_data.update({ aux["word"]: 0 })
                
                    if 'new_verbs' in aux.keys():
                        for new in aux["new_verbs"]:
                            if new not in new_verbs:
                                new_verbs.update({ new: 0 })
                else:
                    preprocessed = preprocessed.replace(' ' + token.text + ' ', ' ')
                    if token.lemma_ not in verbList:
                        #lo agregamos
                        verbList.update({ token.lemma_: 1 })
                    else:
                        #ya existe
                        verbList.update({ token.lemma_: verbList[token.lemma_] + 1 })
        # eliminamos posibles cambios de formato
        # ya se termino de procesar la palabra
        # sacamos los posibles entities obtenidos de los -
        if hyphFlag:
            posEntities = HYPHEntity(preprocessed)

            for posEntity in posEntities:
                if posEntity in hyphEntities:
                    hyphEntities.update({posEntity: hyphEntities[posEntity] + 1})
                else:
                    hyphEntities.update({posEntity: 1})
                entities = SaveEntities(posEntity, entities)
        corpus_preprocessed.append(preprocessed)

    # filtramos el dictionary
    filterBy = [' ', '/', '\\', '.', '_', '-', '+', ',', ':']  # filtramos por varios requisitos

    for filterParam in filterBy:
        entities = FilterEntities(entities, filterParam)
    # ordenamos el diccionario
    # sorted(entities.values())

    # quiero ver los datos
    after = SeeData(entities)

    # vamos a eliminar todos los que tengan frecuencia 1
    # a excepcion de los alfanumericos o numericos
    entities = DeleteEntities(entities)

    before = SeeData(entities)

    # eliminamos todos los que son parte de las entidades de los '-'
    # checamos si existen solas o si alguna contiene otra
    repeatedEntities = {}
    for hyphEntity in hyphEntities:
        separatedEntity = hyphEntity.split('-')
        for entity in separatedEntity:
            if entity in entities:
                # si existe por si sola
                # hay que eliminarla
                del entities[entity]
            else:
                # checamos si lo contiene
                for key, value in entities.items():
                    if entity in key:
                        repeatedEntities.update({key: value})
                        # del entities[key]
                # for key in repeatedEntities:
                # del entities[key]
    
    data = {
        "corpus_preprocessed": corpus_preprocessed,
        "corpus_raw": corpus_raw,
        "entities": entities,
        "after": after,
        "before": before,
        "hyphEntities": hyphEntities,
        "repeatedEntities": repeatedEntities,
        "verbList": verbList,
        "verbs": verbs,
        "new_verbs": new_verbs,
        "word_data": word_data
    }

    return data

def CleaningCorporaThree(nlp, file, vocab_size, base_path):
    nlp = en_core_web_sm.load()
    corpus_preprocessed = []  # texto preprocesado
    corpus_raw = []  # texto original
    entities = {}  # xxx potenciales
    hyphEntities = {}
    greetings = {}  #dic con greetings
    farewells = {}  #dic con farewells
    
    #lista de verbos
    text = FetchInfo(base_path + "Datasets/greetings.txt")
    text_list = text.read().split('\n')
    for i in range(0, len(text_list) - 1):
        greetings.update({ text_list[i]: 0})
    
    text = FetchInfo(base_path + "Datasets/farewells.txt")
    text_list = text.read().split('\n')
    for i in range(0, len(text_list) - 1):
        farewells.update({ text_list[i]: 0})

    for index in range(0, vocab_size):
        corpus_raw.append(file.iat[index, 0])
        #checamos si tiene un greeting
        preprocessed = RemoveGreetings(corpus_raw[index], greetings)
        #checamos si tiene un farewell
        preprocessed = RemoveFarewells(preprocessed, farewells)
        
        preprocessed = RemoveImperfections(preprocessed)
        sentence = nlp(preprocessed)

        i = 0  # contadores auxiliares para encontrar parejas NNP
        hyphFlag = False
        # POS Tagger
        NPPwords = []
        for token in sentence:
            # se agrega uno a i aunque no sea NNP
            i = i + 1
            # checamos otro tipo de token
            if token.text == '-':
                hyphFlag = True
            else:
                text = token.text
                if text[len(text) - 1] == ':':
                    try:
                        # el token que sigue será la entity
                        wordToken = RemoveWhiteSpaces(sentence[i].text.lower())
                        NPPwords.append(wordToken)
                        # entities = SaveEntities(sentence[i + 1].text.lower(), entities)
                    except IndexError:
                        pass
            # token.tag_ == 'HYPH'
            if token.tag_ == 'NNP' and token.text != '-':
                # checamos que no sea el primero
                if i == len(sentence):
                    # checamso que no sea la del después de los :
                    if len(NPPwords) >= 1:
                        wordToken = RemoveWhiteSpaces(token.text.lower())
                        if NPPwords[len(NPPwords) - 1] == wordToken:
                            # lo eliminamos del arreglo, al cabo se vuelve a agregar
                            # esto es para que no se repita
                            NPPwords.pop()
                    # grabamos el NNP en el arreglo
                    NPPword = RemoveWhiteSpaces(token.text.lower())
                    NPPwords.append(NPPword)

                    # es la última, entonces se graba
                    NPPword = ''
                    if len(NPPwords) > 1:
                        NPPword = NPPwords[len(NPPwords) - 1] + ' '
                        NPPwords.pop()
                        for word in reversed(NPPwords):
                            NPPword += word + ' '
                    else:
                        NPPword = NPPwords[0]

                    NPPword = RemoveWhiteSpaces(NPPword)

                    if len(NPPword) > 0:
                        # mandamos como token los NPPs
                        entities = SaveEntities(NPPword, entities)

                    # limpiamos el arreglo
                    NPPwords = []
                else:
                    # es el primero
                    NPPword = RemoveWhiteSpaces(token.text.lower())
                    NPPwords.append(NPPword)
            else:
                # no es NPP
                # grabamos los NPPwords que llegamos
                if len(NPPwords) != 0:
                    NPPword = ''
                    if len(NPPwords) > 1:
                        NPPword = NPPwords[len(NPPwords) - 1] + ' '
                        NPPwords.pop()
                        for word in reversed(NPPwords):
                            NPPword += word + ' '
                    else:
                        NPPword = NPPwords[0]

                    NPPword = RemoveWhiteSpaces(NPPword)
                    if len(NPPword) > 0:
                        # mandamos como token los NPPs
                        entities = SaveEntities(NPPword, entities)
                    # limpiamos el arreglo
                    NPPwords = []
                # else no hay nada que grabar
        # eliminamos posibles cambios de formato
        # ya se termino de procesar la palabra
        # sacamos los posibles entities obtenidos de los -
        if hyphFlag:
            posEntities = HYPHEntity(preprocessed)

            for posEntity in posEntities:
                if posEntity in hyphEntities:
                    hyphEntities.update({posEntity: hyphEntities[posEntity] + 1})
                else:
                    hyphEntities.update({posEntity: 1})
                entities = SaveEntities(posEntity, entities)
        corpus_preprocessed.append(preprocessed)

    # filtramos el dictionary
    filterBy = [' ', '/', '\\', '.', '_', '-', '+', ',', ':']  # filtramos por varios requisitos

    for filterParam in filterBy:
        entities = FilterEntities(entities, filterParam)
    # ordenamos el diccionario
    # sorted(entities.values())

    # quiero ver los datos
    after = SeeData(entities)

    # vamos a eliminar todos los que tengan frecuencia 1
    # a excepcion de los alfanumericos o numericos
    entities = DeleteEntities(entities)

    before = SeeData(entities)

    # eliminamos todos los que son parte de las entidades de los '-'
    # checamos si existen solas o si alguna contiene otra
    repeatedEntities = {}
    for hyphEntity in hyphEntities:
        separatedEntity = hyphEntity.split('-')
        for entity in separatedEntity:
            if entity in entities:
                # si existe por si sola
                # hay que eliminarla
                del entities[entity]
            else:
                # checamos si lo contiene
                for key, value in entities.items():
                    if entity in key:
                        repeatedEntities.update({key: value})
                        # del entities[key]
                # for key in repeatedEntities:
                # del entities[key]
    
    data = {
        "corpus_preprocessed": corpus_preprocessed,
        "corpus_raw": corpus_raw,
        "entities": entities,
        "after": after,
        "before": before,
        "hyphEntities": hyphEntities,
        "repeatedEntities": repeatedEntities
    }

    return data

def CleaningCorporaFour(nlp, file, vocab_size):
    nlp = en_core_web_sm.load()
    corpus_preprocessed = []  # texto preprocesado
    corpus_raw = []  # texto original
    entities = {}  # xxx potenciales
    hyphEntities = {}
    removableTags = ['ADJ', 'DET', 'ADV', 'PART', 'ADP', 'SCONJ', 'SYM', 'INTJ']

    for index in range(0, vocab_size):
        corpus_raw.append(file.iat[index, 0])
        preprocessed = RemoveImperfections(corpus_raw[index])
        sentence = nlp(preprocessed)

        i = 0  # contadores auxiliares para encontrar parejas NNP
        hyphFlag = False
        # POS Tagger
        NPPwords = []
        for token in sentence:
            # se agrega uno a i aunque no sea NNP
            i = i + 1
            # checamos otro tipo de token
            if token.text == '-':
                hyphFlag = True
            else:
                text = token.text
                if text[len(text) - 1] == ':':
                    try:
                        # el token que sigue será la entity
                        wordToken = RemoveWhiteSpaces(sentence[i].text.lower())
                        NPPwords.append(wordToken)
                        # entities = SaveEntities(sentence[i + 1].text.lower(), entities)
                    except IndexError:
                        pass
            # token.tag_ == 'HYPH'
            if token.tag_ == 'NNP' and token.text != '-':
                # checamos que no sea el primero
                if i == len(sentence):
                    # checamso que no sea la del después de los :
                    if len(NPPwords) >= 1:
                        wordToken = RemoveWhiteSpaces(token.text.lower())
                        if NPPwords[len(NPPwords) - 1] == wordToken:
                            # lo eliminamos del arreglo, al cabo se vuelve a agregar
                            # esto es para que no se repita
                            NPPwords.pop()
                    # grabamos el NNP en el arreglo
                    NPPword = RemoveWhiteSpaces(token.text.lower())
                    NPPwords.append(NPPword)

                    # es la última, entonces se graba
                    NPPword = ''
                    if len(NPPwords) > 1:
                        NPPword = NPPwords[len(NPPwords) - 1] + ' '
                        NPPwords.pop()
                        for word in reversed(NPPwords):
                            NPPword += word + ' '
                    else:
                        NPPword = NPPwords[0]

                    NPPword = RemoveWhiteSpaces(NPPword)

                    if len(NPPword) > 0:
                        # mandamos como token los NPPs
                        entities = SaveEntities(NPPword, entities)

                    # limpiamos el arreglo
                    NPPwords = []
                else:
                    # es el primero
                    NPPword = RemoveWhiteSpaces(token.text.lower())
                    NPPwords.append(NPPword)
            else:
                # no es NPP
                # grabamos los NPPwords que llegamos
                if len(NPPwords) != 0:
                    NPPword = ''
                    if len(NPPwords) > 1:
                        NPPword = NPPwords[len(NPPwords) - 1] + ' '
                        NPPwords.pop()
                        for word in reversed(NPPwords):
                            NPPword += word + ' '
                    else:
                        NPPword = NPPwords[0]

                    NPPword = RemoveWhiteSpaces(NPPword)
                    if len(NPPword) > 0:
                        # mandamos como token los NPPs
                        entities = SaveEntities(NPPword, entities)
                    # limpiamos el arreglo
                    NPPwords = []
                # else no hay nada que grabar
            if token.pos_ in removableTags:
                #print(token.text)
                # la eliminamos del texto
                preprocessed = preprocessed.replace(' ' + token.text + ' ', ' ')
        # eliminamos posibles cambios de formato
        # ya se termino de procesar la palabra
        # sacamos los posibles entities obtenidos de los -
        if hyphFlag:
            posEntities = HYPHEntity(preprocessed)

            for posEntity in posEntities:
                if posEntity in hyphEntities:
                    hyphEntities.update({posEntity: hyphEntities[posEntity] + 1})
                else:
                    hyphEntities.update({posEntity: 1})
                entities = SaveEntities(posEntity, entities)
        corpus_preprocessed.append(preprocessed)

    # filtramos el dictionary
    filterBy = [' ', '/', '\\', '.', '_', '-', '+', ',', ':']  # filtramos por varios requisitos

    for filterParam in filterBy:
        entities = FilterEntities(entities, filterParam)
    # ordenamos el diccionario
    # sorted(entities.values())

    # quiero ver los datos
    after = SeeData(entities)

    # vamos a eliminar todos los que tengan frecuencia 1
    # a excepcion de los alfanumericos o numericos
    entities = DeleteEntities(entities)

    before = SeeData(entities)

    # eliminamos todos los que son parte de las entidades de los '-'
    # checamos si existen solas o si alguna contiene otra
    repeatedEntities = {}
    for hyphEntity in hyphEntities:
        separatedEntity = hyphEntity.split('-')
        for entity in separatedEntity:
            if entity in entities:
                # si existe por si sola
                # hay que eliminarla
                del entities[entity]
            else:
                # checamos si lo contiene
                for key, value in entities.items():
                    if entity in key:
                        repeatedEntities.update({key: value})
                        # del entities[key]
                # for key in repeatedEntities:
                # del entities[key]

    data = {
        "corpus_preprocessed": corpus_preprocessed,
        "corpus_raw": corpus_raw,
        "entities": entities,
        "after": after,
        "before": before,
        "hyphEntities": hyphEntities,
        "repeatedEntities": repeatedEntities
    }

    return data

def SplitData(word, verbs, nlp):
    var = {}
    words_to_delete = []
    new_verbs = []
    words = []
    
    filterBy = [' ', '/', '\\', '.', '_', '-', '+', ',', ':']
    
    #paso todos los filter a espacios vacios
    for filterParam in filterBy:
        word = word.replace(filterParam, ' ')
    #quitamos los espacios vacios
    word = RemoveWhiteSpaces(word)
    words = word.split(' ')
    
    for newWord in words:
        #checamos si es parte del dict de verbos
        sentence = nlp(newWord)
        for token in sentence:
            if token.pos_ == 'VERB':
                #checamos si esta en el dict
                if token.lemma_ in verbs:
                    #eliminamos esta
                    words_to_delete.append(token.text)
                    var.update({ "words_to_delete": words_to_delete })
                    word = word.replace(' ' + token.text + ' ', ' ')
                else:
                    #no se encontró un verbo de los que tenemos
                    #agregamos a la lista la palabra completa para ver que parte tiene el verbo
                    new_verbs.append(token.text)
                    var.update({ "new_verbs": new_verbs })
                    word = word.replace(' ' + token.text + ' ', ' ')
                    print('New verb ', token.text)
            else:
                if token.tag_ == 'NNP':
                    print('Possible new entity ', token.text)
    
    #regresamos la nueva palabra separada
    word = RemoveWhiteSpaces(word)
    var.update({ "word": word })
             
    return var


def SaveEntities(token, entities):
    # checamos la existencia del NPP
    if token in entities:
        count = entities[token]
        entities.update({token: count + 1})
    else:
        entities.update({token: 1})

    return entities


def FilterEntities(entities, filterParam):
    # vamos a checar si hay entidades de dos palabras que existen como una
    # quiere decir que son dos nombres distintos (dos aplicaciones)
    aux = entities
    flag = True
    keys = []
    newValues = {}

    for key in entities:
        if filterParam in key:
            # checamos el parametro
            if filterParam is ' ':
                # lo partimos
                words = key.split(filterParam)
                for word in words:
                    # checamos si existe
                    if word not in aux:
                        # PROPUESTA
                        # eliminar la parte que no es NNP
                        # dejamos la palabra compuesta
                        flag = False

                if flag:
                    # si todas existen solas, las partimos
                    for word in words:
                        # le sumamos el count de la palabra sola con la palabra repetida
                        aux.update({word: aux[word] + aux[key]})
                    keys.append(key)
            else:
                # lo partimos
                words = key.split(filterParam)
                for word in words:
                    # checamos si existe
                    if word in aux:
                        # PROPUESTA
                        # hacemos todas las partes, entidades
                        flag = True

                if flag:
                    # si todas existen solas, las partimos
                    for word in words:
                        if word in aux:
                            # le sumamos el count de la palabra sola con la palabra repetida
                            aux.update({word: aux[word] + aux[key]})
                        else:
                            # creamos el nuevo
                            newValues.update({word: 1})
                    # eliminamos el que tenía el separador
                    keys.append(key)

    for value in newValues:
        aux.update({value: 1})

    for key in keys:
        del aux[key]

    return aux


def DeleteEntities(entities):
    aux = entities
    keys = []

    for entity in entities:
        if entities[entity] == 1:
            # checamos que tenga números o solo números
            if not re.match('^(?=.*[0-9]$)', entity) or not re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', entity):
                # este no tiene números o números ni letras
                keys.append(entity)

    for key in keys:
        del aux[key]

    return aux


def HYPHEntity(sentence):
    # pueden ser varios, entonces buscamos todos los '-'
    # ticket = str(sentence.text)
    ticket = sentence
    posEntity = ""
    posEntities = []

    for c in range(ticket.find('-'), len(ticket) - 1):
        if ticket[c] == '-':
            minPos = -1
            maxPos = -1
            # checamos el anterior
            if ticket[c - 1] != ' ':
                # esta pegado al de la izquierda
                if ticket[c + 1] != ' ':
                    # tambien es parte del de la derecha
                    # buscamos el espacio vacio de la izquierda
                    for j in range(c, 0, -1):
                        if ticket[j] == ' ':
                            # llegamos al espacio vacio
                            minPos = j
                            break
                    # buscamos el espacio vacio de la derecha
                    for j in range(c, len(ticket) - 1):
                        if ticket[j] == ' ':
                            # llegamos al espacio vacio
                            maxPos = j
                            break
                    posEntity = ticket[minPos:maxPos]
                    # eliminamos esta entidad del string por si hay más '-'

                else:
                    # solo es parte de la izquierda
                    # buscamos el espacio vacio de la izquierda
                    for j in range(c, 0, -1):
                        if ticket[j] == ' ':
                            # llegamos al espacio vacio
                            minPos = j
                            break
                    posEntity = ticket[minPos:c + 1]
            else:
                # esta separado del de la izquierda
                if ticket[c + 1] != ' ':
                    # es parte del de la derecha
                    # buscamos el espacio vacio de la derecha
                    for j in range(c, len(ticket) - 1):
                        if ticket[j] == ' ':
                            # llegamos al espacio vacio
                            maxPos = j
                            break;
                    posEntity = ticket[c - 1:maxPos]
                # else ninguno debe de llegar aqui (se elimino en preprocesamiento)

        if posEntity != "":
            # lo limpiamos
            posEntity = DeletePunctuation(posEntity.lower())
            # checamos que no estemos tomando el mismo
            # sucede cuando uno tiene dos '-'
            if posEntity not in posEntities:
                posEntities.append(posEntity)
                posEntity = ""

    return posEntities


def DeletePunctuation(text):
    if text.find(' ') == len(text) - 1:
        text = text.replace(' ', '')
    if text.find(' ') == 0:
        text = text.replace(' ', '')
    # ya quitamos los espacios vacios
    if text[len(text) - 1] == ',':
        text = text[:len(text) - 1]
    if text[len(text) - 1] == '.':
        text = text[:len(text) - 1]
    if text[len(text) - 1] == ';':
        text = text[:len(text) - 1]
    # if text[len(text) - 1] == ':':
    # text = text[:len(text) - 1]
    if text[len(text) - 1] == '?':
        text = text[:len(text) - 1]

    return text


def SeeData(entities):
    noneAlpha = []
    alpha = []
    freqOne = []
    alphaInKeys = {}
    alphaNotInKeys = {}

    for entity in entities:
        if entity.isalnum() is False:
            noneAlpha.append(entity)
        if re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', entity):
            alpha.append(entity)
            # tenemos números y letras
            # sacamos los textos que tenga adentro
            # checamos estos textos si existen como keys
            resps = VerifyKeyFromAlphanumeric(entity, entities)

            if len(resps) > 1:
                for resp in resps:
                    if resp:
                        alphaInKeys.update({entity: resp})
                    else:
                        alphaNotInKeys.update({entity: resp})
            else:
                for resp in resps:
                    if resp:
                        alphaInKeys.update({entity: resp})
                    else:
                        alphaNotInKeys.update({entity: resp})

        if entities[entity] == 1:
            freqOne.append(entity)

    arrays = []
    arrays.append(noneAlpha)
    arrays.append(alpha)
    arrays.append(freqOne)
    arrays.append(alphaInKeys)
    arrays.append(alphaNotInKeys)

    return arrays


def VerifyKeyFromAlphanumeric(entity, entities):
    index = -1
    startIndex = -1
    lastIndex = -1
    resp = []

    for c in entity:
        index = index + 1
        if c.isdigit() is False:
            if startIndex is -1:
                startIndex = index
            # else sigue iterando hasta no encontrar string
        else:
            if startIndex is not -1 and lastIndex is -1:
                # ya tenemos un startIndex
                lastIndex = index

                # ya tenemos las dos posiciones
                # cortamos el string y checamos si existe en el dictionario
                text = entity[startIndex:lastIndex]
                if text in entities:
                    # si existe
                    resp.append(True)
                    # reiniciamos los indices para seguir buscando mas strings
                    startIndex = -1
                    lastIndex = -1
                else:
                    resp.append(False)
                    # reiniciamos los indices para seguir buscando mas strings
                    startIndex = -1
                    lastIndex = -1

    # se acaba el loop

    if lastIndex is -1 and startIndex is not -1:
        lastIndex = index
        # ya tenemos las dos posiciones
        # cortamos el string y checamos si existe en el dictionario
        text = entity[startIndex:lastIndex]
        if text in entities:
            # si existe
            resp.append(True)
        else:
            resp.append(False)

    return resp


"""
###################################################################
                      END OF PREPROCESSING 
###################################################################
"""
# %% doc2vec
"""
###################################################################
                            DOC2VEC 
###################################################################
"""


def Doc2VecAlgorithm(corpora, base_path, experiment_name, pretrained_emb, own_model, args, experiment_stage):
    start_time = time.time()
    df_time = corpora['df_time']
    sample = corpora['pipeline']
    sample = sample['corpus_preprocessed']
    vocab_size = corpora['vocab_size']

    parameter_type = int(args['parameter_type'])
    print("save to taggedlinedocument")
    # preprocessed dataset text or csv
    sentences = TaggedLineDocument(
        os.path.join(base_path + "Datasets" + '/' + "PREPROCESS_" + experiment_name + '/' + "dataset.txt"))
    modelSample = TaggedLineDocument(
        os.path.join(base_path + "Datasets" + '/' + "PREPROCESS_" + experiment_name + '/' + "modelSample.txt"))
    trainSample = TaggedLineDocument(
        os.path.join(base_path + "Datasets" + '/' + "PREPROCESS_" + experiment_name + '/' + "trainSample.txt"))
    end_time = time.time()
    df_time = df_time.append(take_time(start_time, end_time, 'taggedlinedocument'), ignore_index=True)

    # getting model
    pretrained_model_files = [base_path + "Models" + '\\' + 'pretrained' + '\\' + 'enwk_dbow', base_path + "Models" + '\\' + 'paper' + '\\' + 'PAPER']
    pretrained_index = 0
    #vamos a hacer pretrained_emb false hasta tener un modelo bien formado
    pretrained_emb = False
    load_model = False
    args['experiment_ref'] = -1 #nose porque hace esto

    """
    # en default pretrained_emb es false y own_model es false
    if pretrained_emb:
        load_model = True
    else:
        if own_model:
            load_model = True
        else:
            load_model = False
            args['experiment_ref'] = -1
    """     
    # own_model
    existing_models = []    #eliminar esta y que se cargue el modelo

    # defined by user
    model_index = 0
    """
    if own_model:
        if args['experiment_ref'] != -1:
            model_name = "Model_" + experiment_stage + str(args['experiment_ref'])
            existing_models.append(model_name)
            print("found your reference")
        else:
            experiment_before_max = args['experiment'] - 1
            for exp_num in range(0, experiment_before_max):
                model_name = "Model_" + experiment_stage + str(exp_num)
                existing_models.append(model_name)
            print("found existing models")
        load_model = True
    else:
        if not pretrained_emb:
            load_model = False
            print('model not found')
    """

            # user defined
    # faster 0, middle 1, accuracy- 2, acccuracy-tuned 3

    # parameters: accuracy 0 vs faster 1, middle 2, accuracy+workers 3
    vector_size_parameter = [100, 200, 300, 300]
    min_count_parameter = [1, 3, 5, 5]
    train_epoch_parameter = [20, 20, 100, 100]
    # needs cyton package to apply
    worker_count_parameter = [10, 5, 1, 10]

    # doc2vec parameters
    vector_size = vector_size_parameter[parameter_type]
    window_size = 15
    min_count = min_count_parameter[parameter_type]
    sampling_threshold = 1e-5
    negative_size = 5
    train_epoch = train_epoch_parameter[parameter_type]
    dm = 0  # 0 = dbow; 1 = dmpv
    worker_count = worker_count_parameter[parameter_type]  # number of parallel processes

    # df_model_parameters=pd.Dataframe({'vector_size:': vector_size,'window size':window_size, 'min_count': min_count, 'sample':sampling_threshold, 'workers':worker_count,'hs':'0' ,'dm':dm, 'negative':negative_size, 'dbow_words':1,'dm_concat':1,'pretrained_emb':pretrained_emb,'iter':train_epoch})
    instantiate = False

    if load_model:
        # which one
        if pretrained_emb:
            model = Doc2Vec.load(pretrained_model_files[pretrained_index])
        else:
            model_loading_path = base_path + "Models" + '\\' + existing_models[model_index]
            if os.path.exists(model_loading_path):
                model = model = Doc2Vec.load(model_loading_path)
            else:
                instantiate = True
                print("no se encontro el modelo")
    else:
        print("instantiate model")
        instantiate = True

    # no debería de estar este en los dos else??
    if instantiate:
        #le damos el 80% de los datos para training
        #modelSample = sentences[:len(sentences) * 0.8]
        cores = multiprocessing.cpu_count()
        print(cores)
        """
        models = [
                    # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
                    Doc2Vec(modelSample, dm=1, dm_concat=1, vector_size=vector_size, window=window_size, negative=5, hs=0, min_count=2, workers=cores, sample=sampling_threshold, epochs=train_epoch),
                    # PV-DBOW 
                    Doc2Vec(modelSample, dm=0, vector_size=vector_size, window=window_size, negative=5, hs=0, min_count=2, dbow_words=1, workers=cores, sample=sampling_threshold, epochs=train_epoch),
                    # PV-DM w/ average
                    Doc2Vec(modelSample, dm=1, dm_mean=1, vector_size=vector_size, window=window_size, negative=5, hs=0, min_count=2, workers=cores, sample=sampling_threshold, epochs=train_epoch),
                ]
        model = Doc2Vec(modelSample, size=vector_size, window=window_size, min_count=2, sample=sampling_threshold,
                        workers=cores, hs=1, dm=1, negative=5, dbow_words=1, dm_concat=1,
                        iter=train_epoch)
        model = Doc2Vec(modelSample, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold,
                        workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1,
                        iter=train_epoch)
        """
        model = Doc2Vec(modelSample, dm=0, vector_size=vector_size, window=window_size, negative=5, hs=0, min_count=2, dbow_words=1, workers=cores, sample=sampling_threshold, epochs=train_epoch)
        #dbow_dmm = ConcatenatedDoc2Vec([models[1], models[2]])
        #dbow_dmc = ConcatenatedDoc2Vec([models[1], models[0]])
        print("Training model")
        #print(dbow_dmm)
        #print(dbow_dmc)
        print(model)
        #trainSample = sentences[len(sentences) * 0.8:]
        #dbow_dmc.train(trainSample, total_examples=int(vocab_size * 0.2), epochs=train_epoch)
        model.train(trainSample, total_examples=int(vocab_size * 0.2), epochs=train_epoch)
        
    """
    # iter = number of iterations (epochs) over the corpus. The default inherited from Word2Vec is 5, but values of 10 or 20 are common in published ‘Paragraph Vector’ experiments.
    # alpha is the initial learning rate (will linearly drop to min_alpha as training progresses).
    if train:
        print("Training model")
        print(model)
        model.train(sentences, total_examples=model.corpus_count, epochs=train_epoch)
        # saving the created model
        # model.save(model_file_path)
    """

    model_doctag = model.docvecs.vectors_docs
    #model_doctag = dbow_dmc.docvecs.models[0].vectors_docs

    end_time_doc2vec = time.time()
    df_time = df_time.append(take_time(start_time, end_time_doc2vec, 'DOC2VECPROCESS'), ignore_index=True)

    resp = {
        "model": model,
        "model_doctag": model_doctag,
        "df_time": df_time
    }

    return resp


"""
###################################################################
                      END OF DOC2VEC 
###################################################################
"""

# %% dimensionality reduction
"""
##################################################################
                DIMENSIONALITY REDUCTION
###################################################################
"""


def PCAalgorithm(DRofW):
    df_time = DRofW['df_time']
    print("PCA")
    start_time = time.time()
    # vectores de doc2vec : output from model
    X = DRofW['model_doctag']

    # 2 dimensions
    pca = PCA(n_components=2)

    # about pca results
    PCA_parameters = pca.get_params()

    # MLE automatic dimensions
    pca2 = PCA(n_components='mle')

    # transform
    model_dim_reduction = pca.fit_transform(X)

    end_time = time.time()
    df_time = df_time.append(take_time(start_time, end_time,'PCA'),ignore_index=True)

    resp = {
        "dim": 2,
        "PCA_parameters": PCA_parameters,
        "pca2": pca2,
        "model_dim_reduction": model_dim_reduction,
        "df_time": df_time
    }

    return resp


def TSNEalgorithm(DRofW):
    df_time = DRofW['df_time']
    print("TSNE")

    start_time = time.time()

    tsne_SCIKIT = TSNE(n_components=2)

    print("Model DocTag Dim Red")

    model_dim_reduction = tsne_SCIKIT.fit_transform(DRofW['model_doctag'])

    print("Finished Model DocTag Dim Red")
    # model_dim_reduction=model_doc_tsne

    end_time = time.time()

    df_time=df_time.append(take_time(start_time, end_time,'TSNE'),ignore_index=True)

    resp = {
        "dim": 2,
        "TSNE_parameters": tsne_SCIKIT.get_params(),
        "model_dim_reduction": model_dim_reduction,
        "df_time": df_time
    }

    return resp


"""
###################################################################
                  END OF DIMENSIONALITY REDUCTION
###################################################################
"""


# %% Silhouette analysis PCA

def ClusteringAlgorithm(dim_reduction, base_path, model_dim_reduction, experiment_name, df_time):
    """
    *******************************************************************

                            SILHOUETTE ANALYSIS

    *******************************************************************
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # print("START evaluacion %s" % datetime.datetime.now())
    start_time = time.time()
    print("Silhouette Analysis")
    # print("START evaluacion %s" % datetime.datetime.now())

    # 0- PCA, 1 TSNE
    if (dim_reduction == 0):
        dim_path = os.path.join(base_path + '/ ' + "Dim Reduction" + '/' + "DIM_REDUCTION_" + experiment_name + '/' + "pca_dim_reduction.csv")
        dim_reduction_name = "PCA"
        range_cluster = 10
    else:
        dim_path = os.path.join(base_path + '/ ' + "Dim Reduction" + '/' + "DIM_REDUCTION_" + experiment_name + '/' + "tsne_dim_reduction.csv")
        dim_reduction_name = "TSNE"
        range_cluster = 50

    with open(dim_path, "w+", encoding='utf-8') as text_file:
        print(dim_reduction_name, ' Evaluation', file=text_file)
        print('Cluster Silhouette Avg ', file=text_file)

        # Copy X_pca into data
        data = model_dim_reduction

        # range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        #                     17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        #                     30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        #                     43, 44, 45, 46, 47, 48, 49, 50]

        k = 0
        avg_Mayor = 0

        for n_clusters in range(2, range_cluster):

            # for n_clusters in range_n_clusters[:10]:
            # TODO: cambiar a 50
            # for n_clusters in range(2,50)

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(data)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(data, cluster_labels)
            print(f'{n_clusters} {silhouette_avg} ', file=text_file)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)
            if (avg_Mayor < silhouette_avg):
                avg_Mayor = silhouette_avg
                k = n_clusters

    print("The ideal number of clusters is ", k,
          " with an average score of ", avg_Mayor)

    # k = k

    end_time = time.time()
    df_time = df_time.append(take_time(start_time, end_time,'SILHOUETTE'),ignore_index=True)
    """
    *******************************************************************

                                CLUSTERING

    *******************************************************************
    """
    
    """
    *******************************************************************

                            K-MEANS SCIKIT LEARN  

    *******************************************************************
    """

    start_time = time.time()

    n_digits = k

    # to change parameters please refer to
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    k_means = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    k_means.fit(data)

    # k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(data, k_means.cluster_centers_)
    labels = k_means.labels_
    print("HERE COMES THE LABELS")
    print(labels)
    CH_Score = metrics.calinski_harabaz_score(data, labels)
    print("HERE COMES THE SCORE")
    print(CH_Score)
    # Plot the centroids as a yellow .
    centroids = k_means.cluster_centers_

    end_time = time.time()
    df_time=df_time.append(take_time(start_time, end_time,'K-MEANS-'+dim_reduction_name),ignore_index=True)
    
    resp = {
        "k_means_labels": k_means_labels,
        "centroids": centroids,
        "data": data,
        "labels": labels,
        "n_digits": n_digits,
        "df_time": df_time
    }

    return resp


# %% summary-results
def SummaryResults(args, DRofW, params, vocab_size, pipeline, KMeansC, df_time):
    # %% write to file-for web
    data = KMeansC['data']
    k = KMeansC['n_digits']
    labels = KMeansC['labels']
    corpus_raw = pipeline['corpus_raw']
    corpus_preprocessed = pipeline['corpus_preprocessed']
    experiment_name = params['experiment_name']
    
    clust_path = os.path.join(
        params['base_path'] + '/' + "Clustering" + '/' + "CLUSTERING_" + experiment_name + '/' + experiment_name + ".txt")
    with open(clust_path, "w+", encoding='utf-8') as text_file:
        print(f'{str(k)} /n ', file=text_file)
        for i_index in range(0, len(data) - 1):
            #print('index ', i_index, data[i_index][0], data[i_index][1], 'labels', labels[i_index], 'corpus_raw', corpus_raw[i_index])
            # print (str(data[i_index][0])+" /n "+str(data[i_index][1])+" /n "+str(labels[i_index])+" /n "+corpus_raw[i_index]+" /n ")
            #print(f'{str(data[i_index][0])} /n {str(data[i_index][1])} /n {str(labels[i_index])} /n {entities[i_index]} /n ', file=text_file)
            print(f'{str(data[i_index][0])} /n {str(data[i_index][1])} /n {str(labels[i_index])} /n {corpus_preprocessed[i_index]} /n {corpus_raw[i_index]} /n ', file=text_file)

    print("clustering_folder")
    print("Clustering")
    print("clust_path")
    print(params['base_path'] + '/' + "Clustering" + '/' + "CLUSTERING_" + experiment_name)
    print(experiment_name + '.txt')
    print(os.path.join(
        params['base_path'] + '/' + "Clustering" + '/' + "CLUSTERING_" + experiment_name+ '/' + experiment_name + ".txt"))
    # time_dict=df_time.to_dict()

    print("*******************************************************************")

    print("summary of results")

    print("Parameters sent")
    print("experiment: ", args['experiment'])
    print("experiment_ref: ", args['experiment_ref'])
    print("parameter_type: ", args['parameter_type'])
    print("model: ", DRofW['model'])
    print("dim_reduction: ", params['dim_reduction'])

    print("Document:", params['doc_name'])
    print("vocabulary size", vocab_size)
    print(experiment_name)
    print("model index", args['parameter_type'])

    print("model attributes:")
    # print(model_attribute_values)
    # end_time_MAIN= time.time()
    # df_time=df_time.append(take_time(start_time_MAIN, end_time_MAIN,'MAIN'),ignore_index=True)

    df_time.to_csv(
        os.path.join(params['base_path'] + '/' + "Metadata" + '/' + "METADATA_" + experiment_name + '/' + "time_log.txt"),
        sep='\t', encoding='utf-8', index=True, header=True)

    print(df_time)


# %% Main
# mandamos llamar todos los métodos
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

df_time = pd.DataFrame(columns=['process', 'total_time'])

start_time_MAIN = time.time()
start_time = time.time()
experiment_name = 0

# definimos los parametros
params = SetParameters(df_time)

# inicializamos los tiempos de preprocesamiento
start_time = time.time()

# preprocesamiento
args = params['args']
corpora = Preprocessing(params['df_time'], params['base_path'], params['experiment_name'], params['args'],
                        params['file_path'], args['pipeline'])

start_time_doc2vec = time.time() 

DRofW = Doc2VecAlgorithm(corpora, params['base_path'], params['experiment_name'], params['pretrained_emb'], params['own_model'], params['args'], 'EXPERIMENT_')

DimRedResults = ""
if(params['dim_reduction'] == 0):
    DimRedResults = PCAalgorithm(DRofW)
else:
    DimRedResults = TSNEalgorithm(DRofW)

KMeansC = ClusteringAlgorithm(params['dim_reduction'], params['base_path'], DimRedResults['model_dim_reduction'], params['experiment_name'], DimRedResults['df_time'])

end_time_MAIN = time.time()
df_time = df_time.append(take_time(start_time_MAIN, end_time_MAIN, 'MAIN'), ignore_index=True)

SummaryResults(params['args'], DRofW, params, corpora['vocab_size'], corpora['pipeline'], KMeansC, df_time)

df_time.to_csv(os.path.join(
    params['base_path'] + '/' + 'Metadata' + '/' + 'METADATA_' + params['experiment_name'] + '/' + 'time_log.txt'),
               sep='\t', encoding='utf-8', index=True, header=True)

print(df_time)