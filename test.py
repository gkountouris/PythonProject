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
    print(('All Data: {}'.format(len(data))))
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
train_path = my_data_path.joinpath('pubmed_factoid_extracted_data_train_triplets_plus_embeddings.p').resolve()
dev_path = my_data_path.joinpath('pubmed_factoid_extracted_data_dev_triplets_plus_embeddings.p').resolve()
test_path = my_data_path.joinpath('pubmed_factoid_extracted_data_test_triplets_plus_embeddings.p').resolve()
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

questions = []
for q, a, s, t, g in train:
    questions.append(q)

print(len(questions))
myset = set(questions)
# find the length of the Python set variable myset
print(len(myset))

data = load_data(train_path, 'factoid_snippet')
questions_Data = []
for q, a, s, t, g in train:
    questions_Data.append(q)

print(len(questions_Data))
myset = set(questions_Data)
# find the length of the Python set variable myset
print(len(myset))

id_keys = []
for key in bioask10:
    for keys in bioask10['questions']:
        id_keys.append(keys['id'])


print(len(id_keys))
myset = set(id_keys)
# find the length of the Python set variable myset
print(len(myset))

id_keys_ours = []
for quests in quest_ids:
    id_keys_ours.append(quests)


print(len(id_keys_ours))
myset = set(id_keys_ours)
# find the length of the Python set variable myset
print(len(myset))

