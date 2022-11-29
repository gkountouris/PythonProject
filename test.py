import json
import pathlib
import pickle
import nltk
from nltk.corpus import stopwords
def load_data(path, keep_only):
    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except:
            data = json.load(f)
    sws = stopwords.words('english')
    ##################################################################################################
    print(('All Data: {}'.format(len(data))))
    ##################################################################################################
    data_ = []
    for (qq, anss, context, type, graph_emb) in data:
        anss_ = []
        if type != keep_only:
            continue
        if len(context) > 500:
            continue
        for ans in anss:
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
            data_.append((qq, anss_, context, type, graph_emb["nodes_original"]))
    ##################################################################################################
    print(('All Data: {}'.format(len(data_))))
    ##################################################################################################
    return data_

my_data_path = pathlib.Path('/home/gk/Documents/BioASQ/BioASQ-data/bioasq_factoid/Graph')

# with open(my_data_path.joinpath('drkg_enhanced_logic_3_200_30_entity_embeddings.json'), 'r') as f:
#   emb = json.load(f)

#
# data = pickle.load(open('/media/gkoun/BioASQ/BioASQ-data/bioasq_factoid/Graph/pubmed_factoid_extracted_data_train_triplets_plus_embeddings.p', 'rb'))
#
#
#
# with open(my_data_path.joinpath('drkg_enhanced_logic_3_200_30_entity_embeddings.json'), 'r') as f:
#     embed = json.load(f)
#
train_path = my_data_path.parent.joinpath('pubmed_factoid_extracted_data.p').resolve()
dev_path = my_data_path.joinpath('pubmed_factoid_extracted_data_dev_triplets_plus_embeddings.p').resolve()
test_path = my_data_path.joinpath('pubmed_factoid_extracted_data_test_triplets_plus_embeddings.p').resolve()

keep_only = 'factoid_snippet'
# train_data = load_data(train_path, keep_only)
dev_data = load_data(dev_path, keep_only)
test_data = load_data(test_path, keep_only)
#
# answered_train = my_data_path.joinpath('pubmed_factoid_extracted_data_valid_answered_perm_train_533_3806.json').resolve()
# with open(my_data_path.joinpath('pubmed_factoid_extracted_data_valid_answered_perm_train_533_3806.json'), 'r') as f:
#   answered_train = json.load(f)
#
with open(my_data_path.parent.parent.joinpath('pmid_to_text_kountouris.json'), 'r') as f:
    pmid = json.load(f)

with open(my_data_path.parent.parent.joinpath('training10b.json'), 'r') as f:
    bioask10 = json.load(f)

with open(my_data_path.parent.parent.joinpath('quest_id_to_pmids_kountouris.json'), 'r') as f:
    quest_ids = json.load(f)

print(train_path)
with open(train_path, 'rb') as f:
  train = pickle.load(f)

keep_only = ['list_random_1', 'list_joint snippets', 'list_before_after_1', 'list_snippet', 'list_before_after_2',
              'list_ideal_answer', 'list_whole abstract', 'list_random_2']

            #, 'factoid_random_2', 'factoid_before_after_2',
             #'factoid_whole abstract', 'factoid_before_after_1', 'factoid_joint snippets',
             #'factoid_random_1', 'factoid_ideal_answer'] # , 'factoid_snippet'
questions = []
types = []
answers = []
for q, a, s, t in train:
    if t in keep_only:
        continue
    answers.append(a)
    questions.append(q)
    types.append(t)
print(set(types))

print('Number of questions we are usings. factoid_snipets: ', len(questions))
using_questions_set = set(questions)
# find the length of the Python set variable myset
print('Number of questions that we are using and are different. factoid_snipets: ', len(using_questions_set))

id_keys = []
type_keys = []
bioask_questions = []
print(bioask10['questions'][0])
for keys in bioask10['questions']:
    if keys['type'] != 'factoid':
        continue
    bioask_questions.append(keys['body'])
    id_keys.append(keys['id'])
    type_keys.append(keys['type'])

#
print(set(type_keys))
#
print('Number of Bioask10 questions id in the Dataset: ', len(id_keys))
myset_bio = set(id_keys)
print('Number of Bioask10 different questions id in the Dataset: ', len(myset_bio))
# find the length of the Python set variable myset
#
print('Number of Bioask10 questions in the Dataset: ', len(bioask_questions))
bioask_questions_set = set(bioask_questions)
print('Number of Bioask10 different questions in the Dataset: ', len(bioask_questions_set))
# find the length of the Python set variable myset
##############################################
from tqdm import tqdm
import os

data = train
final_data = []
for j, entry in tqdm(enumerate(data), total=len(data)):
    if entry[3] in keep_only:
        continue
    id = []
    for keys in bioask10['questions']:
        if keys['type'] != 'factoid':
            continue
        if keys['body'] == entry[0]:
            id = keys['id']
    final_data += [[entry[0], entry[1], entry[2], entry[3], id]]
with open(os.path.join('pubmed_factoid_extracted_data_train_with_id.json'), 'w') as file:
    json.dump(final_data, file)
##############################################
id_keys_ours = []
for quests in quest_ids:
    id_keys_ours.append(quests)

print('Number of questions Dimitris produce: ', len(id_keys_ours))
myset_dimitris = set(id_keys_ours)
# find the length of the Python set variable myset
print('Number of Different questions Dimitris produce: ', len(myset_dimitris))

ids = []
for ids_myset in myset_dimitris:
    for ids_myset_bio in myset_bio:
        if ids_myset == ids_myset_bio:
            ids.append(ids_myset)

print(len(ids))
print(len(set(ids)))

no_id = []

for j, entry in tqdm(enumerate(final_data), total=len(data)):
    if len(entry[4]) < 1:
        no_id.append(j)

print(j)
print(no_id)


from nltk.corpus import stopwords
sws = stopwords.words('english')
data_ = []

count_ans_split = 0

for q, a, s, t, i in final_data:
    anss_ = []
    if t in keep_only:
        continue
    if len(s) > 500:
            continue
    for ans in a:
        if len(ans.strip()) == 0:
            continue
        if len(ans.split()) > 10:
            count_ans_split += 1
            continue
        if ans.lower() in sws:
            continue
        if ans.split()[0].lower() in ['the', 'a']:
            ans = ' '.join(ans.split()[1:])
        anss_.append(ans)
    if len(anss_) > 0:
        data_.append((q, anss_, s, t, i))
print('count_ans_split: ', count_ans_split)
questions = []
types = []
answers = []
for q, a, s, t, i in data_:
    answers.append(a)
    questions.append(q)
    types.append(t)
print(set(types))
print('Number of questions we are usings. factoid_snipets: ', len(questions))
using_questions_set = set(questions)
# find the length of the Python set variable myset
print('Number of questions that we are using and are different. factoid_snipets: ', len(using_questions_set))