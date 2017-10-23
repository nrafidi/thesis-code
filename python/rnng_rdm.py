import numpy as np
import matplotlib
matplotlib.use('Agg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
import string
import Mantel

VECTORS = '/share/volume0/RNNG/sentence_stimuli_tokenized_tagged_pred_trees_no_preterms_vectors.txt'
SENTENCES = '/share/volume0/RNNG/sentence_stimuli_tokenized_tagged_with_unk_final.txt'

NUMAP = 96
WAS = 'was'
BY = 'by'

SEN_LENS = {6: 'A', 8: 'P', 4: 'AS', 5: 'PS'}
KEY_WORDS = {'A': [1, 2, 4], 'P': [1, 3, 6], 'AS': [1, 2], 'PS': [1, 3]}



def syn_rdm(ap_list):
    ap_rdm = np.empty((NUMAP, NUMAP))
    for i, i_sen in enumerate(ap_list):
        if i >= NUMAP:
            break
        for j, j_sen in enumerate(ap_list):
            if j >= NUMAP:
                break
            if i_sen == j_sen:
                ap_rdm[i, j] = 0.0
            elif i_sen in j_sen or j_sen in i_sen:
                ap_rdm[i, j] = 0.5
            else:
                ap_rdm[i, j] = 1.0
    return ap_rdm


def sem_rdm(sen_list):
    ap_rdm = np.empty((NUMAP, NUMAP))
    for i, i_sen in enumerate(sen_list):
        if i >= NUMAP:
            break
        key_words_i = [i_sen[i_word] for i_word in KEY_WORDS[SEN_LENS[len(i_sen)]]]
        print(key_words_i)
        for j, j_sen in enumerate(sen_list):
            if j >= NUMAP:
                break
            key_words_j = [j_sen[j_word] for j_word in KEY_WORDS[SEN_LENS[len(j_sen)]]]
            print(key_words_j)
            print(key_words_j.reverse())
            if i_sen == j_sen:
                ap_rdm[i, j] = 0.0
            elif key_words_i == key_words_j.reverse():
                ap_rdm[i, j] = 0.0
            elif key_words_i == key_words_j:
                ap_rdm[i, j] = 0.0
            elif key_words_i[0] == key_words_j[0] and key_words_i[1] == key_words_j[1]:
                ap_rdm[i, j] = 0.5
            else:
                ap_rdm[i, j] = 1.0
            print(ap_rdm[i, j])
    return ap_rdm


if __name__ == '__main__':
    vectors = np.loadtxt(VECTORS)
    vectors = vectors[:NUMAP, :]

    vec_rdm = squareform(pdist(vectors))

    ap_list = []
    sen_list = []
    with open(SENTENCES) as f:
        for line in f:
            sen_list.append(string.split(line))
            if WAS in line:
                if BY in line:
                    ap_list.append('P')
                else:
                    ap_list.append('PS')
            else:
                if len(string.split(line)) == 4:
                    ap_list.append('AS')
                else:
                    ap_list.append('A')

    ap_rdm = syn_rdm(ap_list)
    semantic_rdm = sem_rdm(sen_list)
    fig, ax = plt.subplots()
    h = ax.imshow(semantic_rdm, interpolation='nearest')
    plt.colorbar(h)
    ax.set_title('Approx Semantic RDM over Active and Passive Sentences')
    fig.savefig('semantic_rdm_actpass.pdf', bbox_inches='tight')

    ktau, pval = kendalltau(vec_rdm, ap_rdm)

    print('Syntactic Kendall tau is {} with pval {}'.format(ktau, pval))

    ktau, pval = kendalltau(vec_rdm, semantic_rdm)

    print('Semantic Kendall tau is {} with pval {}'.format(ktau, pval))

    r, p, z = Mantel.test(vec_rdm, ap_rdm)

    print('Syntactic Pearson is {} with pval {} and zval {} from Mantel test'.format(r, p, z))

    r, p, z = Mantel.test(vec_rdm, semantic_rdm)

    print('Semantic Pearson is {} with pval {} and zval {} from Mantel test'.format(r, p, z))

    fig, ax = plt.subplots()
    h = ax.imshow(vec_rdm, interpolation='nearest')
    plt.colorbar(h)
    ax.set_title('Vector RDM over Active and Passive Sentences')
    fig.savefig('vector_rdm_actpass.pdf', bbox_inches='tight')
    fig, ax = plt.subplots()
    h = ax.imshow(ap_rdm, interpolation='nearest')
    plt.colorbar(h)
    ax.set_title('Approx Syntax RDM over Active and Passive Sentences')
    fig.savefig('syntax_rdm_actpass.pdf', bbox_inches='tight')

    # plt.show()

