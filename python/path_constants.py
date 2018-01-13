import os.path

base_path = '/share/volume0/RNNG'
# base_path = r'C:\Users\danrs\Documents\BrainGroupData\RNNG'

SENTENCES = os.path.join(base_path, 'sentence_stimuli_tokenized_tagged_with_unk_final.txt')

SAVE_MEG_RDM = os.path.join(
    base_path, 'meg_rdm',
    'RDM_{exp}_{word}_{reg}_win{win_size}_avg{avg_time}_{dist}_num{num_instances}_reps{reps_to_use}_proc{proc}{mag}.npz')

SAVE_RDM_SCORES = os.path.join(
    base_path, 'results',
    'Scores_{exp}_{metric}_{reg}_{mode}_{model}_{word}_noUNK{noUNK}_win{win_size}_avg{avg_time}_{dist}_'
    'num{num_instances}_reps{reps_to_use}_proc{proc}{mag}.npz')

semantic_models_path = os.path.join(base_path, 'semantic_models')
wordnet_path = os.path.join(semantic_models_path, 'wordnet')
wordnet_sentence_similarity_path = os.path.join(wordnet_path, 'sentence_similarity', 'direct_sentence_distance')

HUMAN_WORDNET_SEN_RDM = os.path.join(
    wordnet_sentence_similarity_path,
    '{experiment}_{model}_semantic_dissimilarity.npz')

WORDNET_SEQ_RDM = os.path.join(
    wordnet_sentence_similarity_path, '{experiment}_sequential_wordnet_rdm.npz')

WORDNET_HIER_RDM = os.path.join(
    wordnet_sentence_similarity_path, '{experiment}_hierarchical_wordnet_rdm.npz')

WORDNET_WORD_RDM = os.path.join(
    wordnet_path, '{experiment}_all_{word}_RDM_wordnet.npz')
SEMANTIC_VECTORS = os.path.join(semantic_models_path, 'nouns_verb.pkl')
RNNG_VECTORS = os.path.join(base_path, 'sentence_stimuli_tokenized_tagged_pred_trees_no_preterms_vectors.txt')
LSTM_VECTORS = os.path.join(base_path, 'test_sents_vectors_lstm.txt')