import json
import pathlib

my_data_path = pathlib.Path('/home/gk/Documents/BioASQ/BioASQ-data/bioasq_factoid/Graph')

with open(my_data_path.joinpath('drkg_enhanced_logic_3_200_30_entity_embeddings.json'), 'r') as f:
  emb = json.load(f)
#
# data = pickle.load(open('/media/gkoun/BioASQ/BioASQ-data/bioasq_factoid/Graph/pubmed_factoid_extracted_data_train_triplets_plus_embeddings.p', 'rb'))
#
#
#
# with open(my_data_path.joinpath('drkg_enhanced_logic_3_200_30_entity_embeddings.json'), 'r') as f:
#     embed = json.load(f)
#
# train_path = my_data_path.joinpath('pubmed_factoid_extracted_data_train_triplets_plus_embeddings.p').resolve()
# dev_path = my_data_path.joinpath('pubmed_factoid_extracted_data_dev_triplets_plus_embeddings.p').resolve()
# test_path = my_data_path.joinpath('pubmed_factoid_extracted_data_test_triplets_plus_embeddings.p').resolve()
#
# answered_train = my_data_path.joinpath('pubmed_factoid_extracted_data_valid_answered_perm_train_533_3806.json').resolve()
# with open(my_data_path.joinpath('pubmed_factoid_extracted_data_valid_answered_perm_train_533_3806.json'), 'r') as f:
#   answered_train = json.load(f)
#
# with open(my_data_path.parent.parent.joinpath('pmid_to_text_kountouris.json'), 'r') as f:
#     pmid = json.load(f)

# with open(my_data_path.parent.parent.joinpath('training10b.json'), 'r') as f:
#     bioask10 = json.load(f)
#
# # with open(my_data_path.parent.parent.joinpath('quest_id_to_pmids_kountouris.json'), 'r') as f:
# #     quest_ids = json.load(f)
#
#
#
# print(bioask10)



