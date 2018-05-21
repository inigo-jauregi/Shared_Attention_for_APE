import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description='attention_visualize.py')

parser.add_argument('-src', required=True,
                    help='Path to the src sentence file')
parser.add_argument('-mt', required=True,
                    help='Path to the mt sentence file')
parser.add_argument('-pe', required=True,
                    help='Path to the pe predicted sentence file')
parser.add_argument('-attn_matrix', required=True,
                    help='Path to the attention matrix')

opt = parser.parse_args()


def samplemat(dims):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = i
    return aa

the_matrix=pickle.load(open(opt.attn_matrix,'rb'))

#Read each sentence
src_file=open(opt.src,'r',encoding='utf-8')
mt_file=open(opt.mt,'r',encoding='utf-8')
pe_file=open(opt.pe,'r',encoding='utf-8')

for src_line in src_file:
    src_line_splited=src_line.split()
for mt_line in mt_file:
    mt_line_splited=mt_line.split()
for pe_line in pe_file:
    pe_line_splited=pe_line.split()

x_text=pe_line_splited+["<eos>"]
y_text=src_line_splited+mt_line_splited

# Display matrix
plt.matshow(the_matrix)
ticks_x=np.arange(0,len(x_text),1)
ticks_y=np.arange(0,len(y_text),1)
plt.xticks(ticks_x)
plt.yticks(ticks_y)


ax=plt.gca()
ax.set_xticklabels(x_text,rotation='vertical')
ax.set_yticklabels(y_text)

plt.show()
