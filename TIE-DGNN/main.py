import time
import argparse
import pickle
from model import *
from utils import *


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='diginetica/Nowplaying/Tmall/sample/RetailRocket/Gowalla/Lastfm')
parser.add_argument('--hiddenSize', type=int, default=275)
parser.add_argument('--hiddenSize_len', type=int, default=1)
parser.add_argument('--hiddenSize_pos', type=int, default=5)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate.')# [0.001, 0.0005]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--conbeta', type=float, default=0.005, help='conbeta')
parser.add_argument('--corDecay', type=int, default=5,help='Distance Correlation Weight')
parser.add_argument('--DG_dropout', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--DL_dropout', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--numcuda', type=int, default=0,help='which GPU train')
opt = parser.parse_args()
opt.layer_num = 2
opt.in_dims = [275,275]#，每一层输入的hiddensize 无论如何都要保证每一层的hiddensize相同
opt.global_channels = [5,5]
opt.c_dims_global = [55,55]



def main():
    init_seed(2020)
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.numcuda)
    if opt.dataset == 'Nowplaying':
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
    elif opt.dataset == 'Lastfm':
        num_node =  38616
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.3    
    
    else:
        num_node = 310

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))


    adj_in = pickle.load(open('datasets/' + opt.dataset + '/adj_in_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num_in = pickle.load(open('datasets/' + opt.dataset + '/weight_in_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    pos_in = pickle.load(open('datasets/' + opt.dataset + '/pos_in_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    adj_out = pickle.load(open('datasets/' + opt.dataset + '/adj_out_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num_out = pickle.load(open('datasets/' + opt.dataset + '/weight_out_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    pos_out = pickle.load(open('datasets/' + opt.dataset + '/pos_out_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    adj_io = pickle.load(open('datasets/' + opt.dataset + '/adj_io_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num_io = pickle.load(open('datasets/' + opt.dataset + '/weight_io_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    adj_in,num_in,pos_in=handle_adj_(adj_in,num_node,opt.n_sample,num_in,pos_in)
    adj_out, num_out, pos_out = handle_adj_(adj_out, num_node, opt.n_sample, num_out, pos_out)
    adj_io,num_io=handle_adj(adj_io,num_node,opt.n_sample,num_io)
    adj_all=[adj_in,adj_out,adj_io]
    num_all=[num_in,num_out,num_io]
    pos_all=[pos_in,pos_out]

    train_data = Data(train_data)
    test_data = Data(test_data)

    model = trans_to_cuda(CombineGraph(opt, num_node, adj_all, num_all,pos_all))
    Cor_loss_model = trans_to_cuda(Cor_loss(opt.corDecay, opt.global_channels[-1], opt.in_dims[-1], opt.c_dims_global[-1]))
    print(opt)
    start = time.time()

    top_K = [5,10, 20]
    best_result={}
    best_epoch={}
    for K in top_K:
        best_result['@%d' % K] = [0,0]
        best_epoch['@%d' % K] = [0,0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics = train_test(model, train_data, test_data,Cor_loss_model)
        for i in top_K:
            hit=metrics['hit%d' % i]
            mrr=metrics['mrr%d' % i]
            flag = 0
            if hit >= best_result['@%d' % i][0]:
                best_result['@%d' % i][0] = hit
                best_epoch['@%d' % i][0] = epoch
                flag = 1
            if mrr >= best_result['@%d' % i][1]:
                best_result['@%d' % i][1] = mrr
                best_epoch['@%d' % i][1] = epoch
                flag = 1
            print('Result@%d' % i )
            print('Current Result@%d:\tRecall@%d:\t%.4f\tMMR%d:\t%.4f' % (i,i,hit, i,mrr))
            print('Best Result@%d:\tRecall@%d:\t%.4f\tMMR%d:\t%.4f\tEpoch:\t%d,\t%d' % (i,i,
                best_result['@%d' % i][0], i,best_result['@%d' % i][1], best_epoch['@%d' % i][0], best_epoch['@%d' % i][1]))

            bad_counter += 1 - flag
            if bad_counter >= opt.patience:
                break

    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
