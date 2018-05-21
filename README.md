# A Shared Attention Mechanism for Better Interpretability of Neural Automatic Post-Editing Systems

This is a variation of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system, that implements the paper ''A Shared Attention Mechanism for Better Interpretability of Neural Automatic Post-Editing Systems" (link). The code is in [Pytorch](https://github.com/pytorch/pytorch).

## Quick-start
The following file describe how to run a basic model. Other hyper-parameters are available. Look at the _opt.py_ file.

1. Preprocess the data:

'''
python preprocess.py -train_src <train_src> -train_inter <train_mt> -train_tgt <train_pe> -valid_src <val_src> -valid_inter <val_mt> -valid_tgt <val_pe> -save_data <output> -lower
'''

2. Train the model:

'''
python train.py -encoder_type double_encoder -data <preprocesed_file> -save_model <output_path> -word_vec_size <vector_dimension> -epochs <num_epochs>
'''

3. Translate sentences:

'''
python translate.py -model <path_to_trained_model> -src <test_src> -inter <test_mt> -output <prediction_file> -replace_unk
'''

## EXTRA TOOLS

Compute BLEU score:

'''
perl tools/multi-bleu.perl    <reference_file>     <    <predicted_file>
'''

Merge back subword-units:

'''
sed -r 's/(@@ )|(@@ ?$)//g'    <pred_file_in_subword_units>   >     <merged_subword_units_output>
'''

*Note that the subword units learned with Byte Pair Encoding in our paper are provided in "subword_BPE/en_de_bpe.30000"

Attention visualization:

In order to visualize the attention, you need to store the src, mt and predicted pe sentence in diferent files (one sentence in each file). Moreover, you need to save the attention matrix predicted by the model in that example with the option -save_attention in the translate.py file. Then you can run this script:

'''
python   tools/attention_visualize.py   -src   <path_to_src_sentence>   -mt   <path_to_the_mt_sentence>   -pe   <path_to_the_predicted_pe_sentence>   -attn_matrix   <path_to_the_predicted_attention_matrix>
'''
