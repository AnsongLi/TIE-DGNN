import pickle
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='diginetica/Tmall/Nowplaying/sample')
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num
t=time.time()
print('建图开始')
seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))


if dataset == "Tmall":
    num = 40728
elif dataset == "Nowplaying":
    num = 60417
elif dataset == 'Lastfm':
    num =  38616
else:
    num = 310

relation_in = []

relation_out=[]
relation_pos=[]
neighbor = [] * num

all_test = set()

adj_in_dict = [dict() for _ in range(num)]
adj_out_dict = [dict() for _ in range(num)]
adj_io_dict = [dict() for _ in range(num)]
adj_in = [[] for _ in range(num)]
adj_out = [[] for _ in range(num)]
adj_io = [[] for _ in range(num)]


for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 4):
        for j in range(len(data)-k):
            relation_out.append([data[j], data[j+k]])
            relation_in.append([data[j+k], data[j]])
            relation_pos.append(k)



for tup,pos in zip(relation_in,relation_pos):
    if tup[1] in adj_in_dict[tup[0]].keys():
        adj_in_dict[tup[0]][tup[1]].append(pos)
    else:
        adj_in_dict[tup[0]][tup[1]] = [pos]

for tup,pos in zip(relation_out,relation_pos):
    if tup[1] in adj_out_dict[tup[0]].keys():
        adj_out_dict[tup[0]][tup[1]].append(pos)
    else:
        adj_out_dict[tup[0]][tup[1]] = [pos]


for i in range(1,len(adj_out_dict)):
    if adj_out_dict[i].keys()&adj_in_dict[i].keys() ==  set():
        pass
    else:
        for j in adj_out_dict[i].keys()&adj_in_dict[i].keys():
            adj_io_dict[i][j] = len(adj_in_dict[i][j])+len(adj_out_dict[i][j])
            adj_in_dict[i].pop(j)
            adj_out_dict[i].pop(j)


weight_in = [[] for _ in range(num)]
pos_in=[[] for _ in range(num)]
weight_out = [[] for _ in range(num)]
pos_out=[[] for _ in range(num)]
weight_io = [[] for _ in range(num)]


for t in range(num):
    x = [v for v in sorted((adj_in_dict[t].items()), reverse=True, key=lambda x: len(x[1]))]
    adj_in[t] = [v[0] for v in x]
    weight_in[t] = [len(v[1]) for v in x]
    pos_in[t]=[ max(v[1],key=v[1].count) for v in x]

    y = [v for v in sorted((adj_out_dict[t].items()), reverse=True, key=lambda x: len(x[1]))]
    adj_out[t] = [v[0] for v in y]
    weight_out[t] = [len(v[1]) for v in y]
    pos_out[t]=[ max(v[1],key=v[1].count) for v in y]

    z = [v for v in sorted((adj_io_dict[t].items()), reverse=True, key=lambda x: x[1])]
    adj_io[t] = [v[0] for v in z]
    weight_io[t] = [(v[1]) for v in z]



for i in range(num):
    adj_in[i] = adj_in[i][:sample_num]
    weight_in[i] = weight_in[i][:sample_num]
    pos_in[i] = pos_in[i][:sample_num]

    adj_out[i] = adj_out[i][:sample_num]
    weight_out[i] = weight_out[i][:sample_num]
    pos_out[i] = pos_out[i][:sample_num]

    adj_io[i] = adj_io[i][:sample_num]
    weight_io[i] = weight_io[i][:sample_num]

pickle.dump(adj_in, open('datasets/' + dataset + '/adj_in_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight_in, open('datasets/' + dataset + '/weight_in_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(pos_in, open('datasets/' + dataset + '/pos_in_' + str(sample_num) + '.pkl', 'wb'))


pickle.dump(adj_out, open('datasets/' + dataset + '/adj_out_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight_out, open('datasets/' + dataset + '/weight_out_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(pos_out, open('datasets/' + dataset + '/pos_out_' + str(sample_num) + '.pkl', 'wb'))

pickle.dump(adj_io, open('datasets/' + dataset + '/adj_io_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight_io, open('datasets/' + dataset + '/weight_io_' + str(sample_num) + '.pkl', 'wb'))


t1=time.time()
print("Run time: %f s" % ((t1 - t)/3600))