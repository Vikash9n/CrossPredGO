import numpy as np
import pandas as pd
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.layers import Embedding, RepeatVector, Permute
from tensorflow.keras.layers import Convolution1D, MaxPooling1D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Flatten, BatchNormalization,Lambda
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Dense, Dropout, Activation, Input, Multiply,Flatten, BatchNormalization,Lambda)
from tensorflow.keras.layers import concatenate
from collections import deque 
from tensorflow.keras.layers import maximum,LSTM,Bidirectional,Reshape,GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop,Adam,SGD,Adagrad
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from tensorflow.keras.layers import MultiHeadAttention 
from tensorflow.keras import backend as K

#Load Data into Train, Test, Valid Datasets

def load_data(df,test_df,org=None):
    n = len(df)
    index = df.index.values
    valid_n = int(n * 0.8)
    train_df = df.loc[index[:valid_n]]
    valid_df = df.loc[index[valid_n:]]
    if org is not None:
        logging.info('Unfiltered test size: %d' % len(test_df))
        test_df = test_df[test_df['orgs'] == org]
        logging.info('Filtered test size: %d' % len(test_df))
    def reshape(values):
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    def normalize_minmax(values):
        mn = np.min(values)
        mx = np.max(values)
        if mx - mn != 0.0:
            return (values - mn) / (mx - mn)
        return values - mn

    def get_values(data_frame):
        labels = reshape(data_frame['labels'].values)
        ngrams = sequence.pad_sequences(
            data_frame['ngrams'].values, maxlen=MAXLEN)
        ngrams = reshape(ngrams)
        rep = reshape(data_frame['struct_feature'].values)
        emb = reshape(data_frame['embeddings'].values)
        data = (ngrams, rep ,emb)
        return data, labels

    train = get_values(train_df)
    valid = get_values(valid_df)
    test = get_values(test_df)

    return train,valid,test,train_df,valid_df,test_df
	
#Gene Ontology Information

def get_gene_ontology(filename='go.obo'):
    go = dict()
    obj = None
    with open('Multi-PredGO-master/data/' + filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        go[obj['id']] = obj
    for go_id in list(go.keys()):
        if go[go_id]['is_obsolete']:
            del go[go_id]
    for go_id, val in go.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in go:
                if 'children' not in go[p_id]:
                    go[p_id]['children'] = set()
                go[p_id]['children'].add(go_id)
    return go

#Get Sequence feature models
def get_feature_model(params):
    embedding_dims = params['embedding_dims']
    max_features = 8001
    model = Sequential()
    model.add(Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN))
    model.add(Dropout(params['embedding_dropout'])) 
    model.add(Bidirectional(GRU(128)))
    model.summary()
    return model	

#Hierachial classification
def get_node_name(go_id, unique=False):
    name = go_id.split(':')[1]
    if not unique:
        return name
    if name not in node_names:
        node_names.add(name)
        return name
    i = 1
    while (name + '_' + str(i)) in node_names:
        i += 1
    name = name + '_' + str(i)
    node_names.add(name)
    return name

def get_function_node(name, inputs):
    output_name = name + '_out'
    net = Dense(256, name=name, activation='relu')(inputs)
    output = Dense(1, name=output_name, activation='sigmoid')(inputs)
    return net, output

def get_parents(go, go_id):
    go_set = set()
    for parent_id in go[go_id]['is_a']:
        if parent_id in go:
            go_set.add(parent_id)
    return go_set

def get_layers(inputs,func_set,functions,go_group):
    BIOLOGICAL_PROCESS = 'GO:0008150'
    MOLECULAR_FUNCTION = 'GO:0003674'
    CELLULAR_COMPONENT = 'GO:0005575'
    FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS
    }
    go=get_gene_ontology()
    GO_ID=FUNC_DICT[go_group]
    q = deque()
    layers = {}
    name = get_node_name(GO_ID)
    layers[GO_ID] = {'net': inputs}
    for node_id in go[GO_ID]['children']:
        if node_id in func_set:
            q.append((node_id, inputs))
    while len(q) > 0:
        node_id, net = q.popleft()
        parent_nets = [inputs]
        name = get_node_name(node_id)
        net, output = get_function_node(name, inputs)
        if node_id not in layers:
            layers[node_id] = {'net': net, 'output': output}
            for n_id in go[node_id]['children']:
                if n_id in func_set and n_id not in layers:
                    ok = True
                    for p_id in get_parents(go, n_id):
                        if p_id in func_set and p_id not in layers:
                            ok = False
                    if ok:
                        q.append((n_id, net))

    for node_id in functions:
        childs = set(go[node_id]['children']).intersection(func_set)
        if len(childs) > 0:
            outputs = [layers[node_id]['output']]
            for ch_id in childs:
                outputs.append(layers[ch_id]['output'])
            name = get_node_name(node_id) + '_max'
            layers[node_id]['output'] = maximum(outputs,name=name)
    return layers

#Train Model Function
def train_model(go_group,params):
  func_df = pd.read_pickle("Multi-PredGO-master/data/"+go_group + '.pkl')
  functions = func_df['functions'].values
  func_set = set(functions)
  M_inputs = Input(shape=(MAXLEN,), dtype='int32', name='inputs')  #(None, 256) sequence
  M_inputs1 = Input(shape=(REPLEN,), dtype='float32', name='input1') #(None, 256) 3d structure
  M_inputs2 = Input(shape=(REPLEN,), dtype='float32', name='input2') #(None, 256) ppi
  feature_model = get_feature_model(params)(M_inputs)

  feature_model = Reshape((1,256))(feature_model)
  inputs1 = Reshape((1,256))(M_inputs1)
  inputs2 = Reshape((1,256))(M_inputs2)

  net_feature = concatenate([feature_model,inputs1,inputs2],axis=1, name='net_O_feature')

  # compute importance for each step
  attention = Dense(1, activation='tanh')(net_feature)
  attention = Flatten()(attention)
  attention = Activation('softmax')(attention)
  attention = RepeatVector(256)(attention)
  attention = Permute([2, 1])(attention)

  sent_representation = Multiply()([net_feature, attention])
  sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
  net = Dropout(0.4)(sent_representation)
  net = Dense(256)(net)

  layers = get_layers(net,func_set,functions,go_group)
  output_models = []
  for i in range(len(functions)):
    output_models.append(layers[functions[i]]['output'])
  net =  concatenate(output_models,axis=1)
  net = Dense(1024, activation='relu')(net)
  net = Dense(len(functions), activation='sigmoid')(net)
  model = Model(inputs=[M_inputs,M_inputs1,M_inputs2], outputs=net)
  optimizer = RMSprop()
  model.compile(optimizer=optimizer,loss='binary_crossentropy')
  model.summary()
  return model
	
#Compute Performance
def get_go_set(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set

def get_anchestors(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        for parent_id in go[g_id]['is_a']:
            if parent_id in go:
                q.append(parent_id)
    return go_set


def compute_performance(preds, labels, gos,go_group):
    func_df = pd.read_pickle("Multi-PredGO-master/data/"+go_group + '.pkl')
    functions = func_df['functions'].values
    func_set = set(functions)
    BIOLOGICAL_PROCESS = 'GO:0008150'
    MOLECULAR_FUNCTION = 'GO:0003674'
    CELLULAR_COMPONENT = 'GO:0005575'
    FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS
    }
    go=get_gene_ontology()
    GO_ID=FUNC_DICT[go_group]
    all_functions = get_go_set(go, GO_ID)
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            all_gos = set()
            for go_id in gos[i]:
                if go_id in all_functions:
                    all_gos |= get_anchestors(go, go_id)
            all_gos.discard(GO_ID)
            all_gos -= func_set
            fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max
	
#Compute ROC and MCC Score
def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(preds, labels):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

#Hyper Parameter Setting
global MAXLEN,batch_size,n_epoch,REPLEN,go_group
MAXLEN=1000
batch_size=int(50)
n_epoch=int(100)
REPLEN= 256
go_group='mf' # 'cc', 'bp' for other ontology
params = {
            'learning_rate': 0.001,
            'embedding_dims': 128,
            'embedding_dropout': 0.2,
            'nb_dense': 1,
        }

#Load Data
train_df=pd.read_pickle('Multi-PredGO-master/data/'+'multimodaltrain-'+go_group+'.pkl')
test_df=pd.read_pickle('Multi-PredGO-master/data/'+'multimodaltest-'+go_group+'.pkl')
train, val, test, train_df, valid_df, test_df = load_data(train_df,test_df)
train_data, train_labels = train
val_data, val_labels = val
test_data, test_labels = test
test_gos = test_df['gos'].values

#Train Model
model_path='Models/Ashish_multi_1_attention_bi-gru_model_4' + go_group + '.h5'
checkpointer = ModelCheckpoint(filepath=model_path,verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model = train_model(go_group,params)
model.fit(x=train_data,y=train_labels,batch_size=batch_size,epochs=n_epoch,validation_data=(val_data,val_labels),callbacks=[checkpointer, earlystopper])

#Predict Model
pred_labels = model.predict(x=test_data)

#Evaluate Performance
f, p, r, t, preds_max = compute_performance(pred_labels, test_labels, test_gos,go_group)
roc_auc = compute_roc(pred_labels, test_labels)
mcc = compute_mcc(preds_max, test_labels)




























