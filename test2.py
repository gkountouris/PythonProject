import json
import nltk
from nltk.corpus import stopwords
import pathlib
import pickle
from tqdm import tqdm
import os
from nltk.corpus import stopwords
sws = stopwords.words('english')
import nltk.data
import time

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


my_data_path = pathlib.Path('/home/gk/Documents/BioASQ/BioASQ-data/bioasq_factoid/Graph')

with open('squad1.1/train-v1.1.json', 'r') as f:
    squad = json.load(f)

related_titles = ['Antibiotics', 'Anthropology', 'Genome', 'Symbiosis', 'Gene', 'Biodiversity', 'Digestion',
                  'Circadian_rhythm', 'Pharmaceutical_industry', 'Myocardial_infarction', 'Botany', 'Pesticide',
                  'Tuberculosis', 'On_the_Origin_of_Species', 'Asthma', 'Diarrhea', 'Pain', 'Bacteria', 'Infection']

squad_data = []
snipet = ''
snipet_minus1 = ''
snipet_plus1 = ''
for data in squad['data']:
    if data['title'] in related_titles:
        for paragraphs in data['paragraphs']:
            context = paragraphs['context']
            sentences = tokenizer.tokenize(context)
            for qas in paragraphs['qas']:
                question = qas['question']
                id = qas['id']
                answer = qas['answers'][0]['text']
                for idx, sentence in enumerate(sentences):
                    if answer in sentence:
                        snipet = sentences[idx]
                    try:
                        snipet_plus1 = sentences[idx + 1]
                    except:
                        pass
                    try:
                        snipet_minus1 = sentences[idx - 1]
                    except:
                        pass
                snipet_plus_minus1 = snipet_minus1 + ' ' + snipet + ' ' + snipet_plus1
                if len(snipet) > 500:
                    continue
                squad_data.append([question, answer, snipet, 'squad', id])
                if len(snipet_plus_minus1) > 500:
                    continue
                squad_data.append([question, answer, snipet_plus_minus1, 'squad', id])

print('SQUAD: ', len(squad_data))

train_path = my_data_path.parent.joinpath('pubmed_factoid_extracted_data.p').resolve()
with open(train_path, 'rb') as f:
  train = pickle.load(f)

with open(my_data_path.parent.parent.joinpath('training10b.json'), 'r') as f:
    bioask10 = json.load(f)

keep_only = ['factoid_before_after_1', 'factoid_before_after_2', 'list_snippet', 'list_before_after_2',
                 'list_before_after_1', 'factoid_snippet']

data = train
data_pubmed = []
for j, entry in tqdm(enumerate(train), total=len(train)):
    id = []
    anss_ = []
    if entry[3] not in keep_only:
        continue
    if len(entry[2]) > 500:
            continue
    for ans in entry[1]:
        if len(ans.strip()) == 0:
            continue
        if len(ans.split()) > 10:
            continue
        if ans.lower() in sws:
            continue
        if ans.split()[0].lower() in ['the', 'a']:
            ans = ' '.join(ans.split()[1:])
        anss_.append(ans)
    if len(anss_) > 0:
        for keys in bioask10['questions']:
            # if keys['type'] != 'factoid':
            #     continue
            if keys['body'] == entry[0]:
                id = keys['id']
        # print(id)
        data_pubmed.append([entry[0], anss_, entry[2], entry[3], id])

print('pubmed: ', len(data_pubmed))

with open('data/COVID-QA.json', 'r') as f:
    covid = json.load(f)

covid_data = []
snipet = ''
snipet_plus1 = ''
snipet_minus1 = ''

for data in covid['data']:
    for paras in data['paragraphs']:
        context = paras['context']
        sentences = tokenizer.tokenize(context)
        for qas in paras['qas']:
            question = qas['question']
            if 'Figure' in question:
                continue
            answer = qas['answers'][0]['text']
            if len(answer.strip()) == 0:
                continue
            if len(answer.split()) > 10:
                continue
            id = qas['id']
            for idx, sentence in enumerate(sentences):
                if answer in sentence:
                    snipet = sentences[idx]
                try:
                    snipet_plus1 = sentences[idx + 1]
                except:
                    pass
                try:
                    snipet_minus1 = sentences[idx - 1]
                except:
                    pass
            if len(snipet) == 0:
                continue
            snipet_plus_minus1 = snipet_minus1 + ' ' + snipet + ' ' + snipet_plus1
            if len(snipet) > 500:
                continue
            covid_data.append([question, answer, snipet, 'covid', id])
            if len(snipet_plus_minus1) > 500:
                continue
            covid_data.append([question, answer, snipet_plus_minus1, 'covid', id])

print(len(covid_data))

final_data = data_pubmed + squad_data + covid_data

print('final: ', len(final_data))

with open(os.path.join('pubmed_squad_covid_data_train.json'), 'w') as file:
    json.dump(final_data, file)

