# import os; os.chdir('C:/Users/ingulbull/Desktop/2019-1/Repro_study_2019_1')
import preprocess as prp

src = {'char_vocab':prp.char_vocab,
       'word_vocab':prp.word_vocab,
       'maxLen':prp.max_len + 2,
       'time_steps':35,
       'embed_size_char':15,
       'embed_size_word':300,
       'num_filter_per_width':25,
       'widths':[1,2,3,4,5,6],
       'hidden_size':300,
       'num_layer':2,
       'batch_size':20
       }
