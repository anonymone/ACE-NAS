import sys
sys.path.append('./')
from nasbench import api
import random
from pandas import DataFrame
import os
from Evaluator.Utils import recoder

nasbench_data_path = './Res/nasbench_full.tfrecord.1' # the script always run on project root

train_set_portion = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
total_samples = 423000
save_path = './Res/nasbench/'
recoder.create_exp_dir(save_path)

operation_token = {
    'input':'2',
    'output':'3',
    'conv3x3-bn-relu': '4',
    'conv1x1-bn-relu' : '5',
    'maxpool3x3' : '6'
}

# load data
nasbench423k = api.NASBench(nasbench_data_path)

# hash values
hash_models = list(nasbench423k.hash_iterator())

value_list = []
for hash_str in hash_models:
    nasbench_value = nasbench423k.get_metrics_from_hash(hash_str)
    matrix, operations, _ = nasbench_value[0].values()
    valid_acc = [nasbench_value[-1][v][-1]['final_validation_accuracy'] for v in nasbench_value[-1].keys()]
    encoding_string = '{0} {1}'.format(' '.join([str(bit) for bit in matrix.reshape(-1).tolist()]), ' '.join([ operation_token[op] for op in operations]))
    value_list.append((hash_str, encoding_string, valid_acc))

# free memory
del nasbench423k
del hash_models

for p in train_set_portion:
    table_train = {
        'Encoding string':[],
        'accuracy 4': [],
        'accuracy 12': [],
        'accuracy 36': [],
        'accuracy 108': [],
        
    }
    table_valid = {
        'Encoding string':[],
        'accuracy 4': [],
        'accuracy 12': [],
        'accuracy 36': [],
        'accuracy 108': [],
        
    }
    random.shuffle(value_list)
    candidates = int(total_samples*p)
    
    m = 1
    n = 0
    for hash_str, encoding_string, valid_acc in value_list[:candidates]:
        # build presave date
        table_train['Encoding string'].append(encoding_string)
        table_train['accuracy 4'].append(valid_acc[0])
        table_train['accuracy 12'].append(valid_acc[1])
        table_train['accuracy 36'].append(valid_acc[2])
        table_train['accuracy 108'].append(valid_acc[3])
        n += 1
        if n == 4230:
            print(m)
            m += 1
            n = 0
    for hash_str, encoding_string, valid_acc in value_list[candidates:]:
        table_valid['Encoding string'].append(encoding_string)
        table_valid['accuracy 4'].append(valid_acc[0])
        table_valid['accuracy 12'].append(valid_acc[1])
        table_valid['accuracy 36'].append(valid_acc[2])
        table_valid['accuracy 108'].append(valid_acc[3])
        n += 1
        if n == 4230:
            print(m)
            m += 1
            n = 0
    # save seq2seq train data
    if p == 0.001:
        print('save seq2seq data...')
        all_encoding = table_train['Encoding string'] + table_valid['Encoding string']
        all_train = ['{0} \t {0}\n'.format(code) for code in all_encoding]
        # save 
        recoder.create_exp_dir(os.path.join(save_path, 'seq2seq'))
        with open(os.path.join(save_path, 'seq2seq', 'train'), mode='w') as file:
            n = 0
            m = 1
            for all_code in all_train:
                file.write(all_code)
                n += 1
                if n == 4230:
                    print('save {0} times'.format(m))
                    file.flush()
                    m += 1
                    n = 0
            file.flush()
    print('save {0} p data...'.format(p))
    table_train = DataFrame(table_train)
    table_valid = DataFrame(table_valid)
    recoder.create_exp_dir(os.path.join(save_path, str(p)))
    table_train.to_csv(os.path.join(save_path, str(p), 'train.csv'))
    table_valid.to_csv(os.path.join(save_path, str(p), 'valid.csv'))



