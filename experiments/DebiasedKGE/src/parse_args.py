import argparse
parser = argparse.ArgumentParser(description='Parser For Arguments',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# training control
parser.add_argument('-snapshot_num', dest='snapshot_num', default=5, help='The snapshot number of the dataset')
parser.add_argument('-dataset', dest='dataset', default='ENTITY', help='dataset name')
parser.add_argument('-gpu', dest='gpu', default=0)
parser.add_argument('-loss_name', dest='loss_name', default='Margin', help='Margin: pairwise margin loss')
parser.add_argument('-train_new', dest='train_new', default=True, help='True: Training on new facts; False: Training on all seen facts')
parser.add_argument('-skip_previous', dest='skip_previous', default='False', help='Allow re-training and snapshot_only models skip previous training')

# model setting
parser.add_argument('-lifelong_name', dest='lifelong_name', default='DebiasedKGE')
parser.add_argument('-optimizer_name', dest='optimizer_name', default='Adam')
parser.add_argument('-embedding_model', dest='embedding_model', default='TransE')
parser.add_argument('-epoch_num', dest='epoch_num', default=200, help='max epoch num')
parser.add_argument('-margin', dest='margin', default=8.0, help='The margin of MarginLoss')
parser.add_argument('-batch_size', dest='batch_size', default=2048, help='Mini-batch size')
parser.add_argument('-learning_rate', dest='learning_rate', default=0.0001)
parser.add_argument('-emb_dim', dest='emb_dim', default=200, help='embedding dimension')
parser.add_argument('-l2', dest='l2', default=0.0, help='optimizer l2')
parser.add_argument('-neg_ratio', dest='neg_ratio', default=10, help='the ratio of negative/positive facts')
parser.add_argument('-patience', dest='patience', default=3, help='early stop step')
parser.add_argument('-regular_weight', dest='regular_weight', default=0.01, help='Regularization strength: alpha')
parser.add_argument('-reconstruct_weight', dest='reconstruct_weight', default=0.1, help='The weight of MAE loss: beta')
parser.add_argument('-new_loss_weight', dest='new_loss_weight', default=0.1, type=float, help='The weight for new loss in the multi-task loss function')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size for the model')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers for GCN')
parser.add_argument("--num_factors", type=int, default=2, help="Number of disentangled factors")
parser.add_argument("--using_disentangled_loss", type=int, default=1, help="Set to 1 to use disentangled loss")
parser.add_argument('--alignment_weight', type=float, default=1, help="Weight for alignment regularization")

# ablation study
parser.add_argument('-using_regular_loss', dest='using_regular_loss', default='True')
parser.add_argument('-using_reconstruct_loss', dest='using_reconstruct_loss', default='True')
parser.add_argument('-using_embedding_transfer', dest='using_embedding_transfer', default='True')
parser.add_argument('-using_finetune', dest='using_finetune', default='True')
parser.add_argument('-using_Knowledge_alignment', dest='using_Knowledge_alignment', default='True')

# others
parser.add_argument('-save_path', dest='save_path', default='./checkpoint/')
parser.add_argument('-data_path', dest='data_path', default='./data/')
parser.add_argument('-log_path', dest='log_path', default='./logs/')
parser.add_argument('-num_layer', dest='num_layer', default=1, help='MAE layer')
parser.add_argument('-num_workers', dest='num_workers', default=1)
parser.add_argument('-valid_metrics', dest='valid_metrics', default='mrr')
parser.add_argument('-valid', dest='valid', default=True, help='indicator of test or valid')
parser.add_argument('-note', dest='note', default='', help='The note of log file name')
parser.add_argument('-seed', dest='seed', default=55, help='random seed, 11 22 33 44 55 for our experiments')
args = parser.parse_args()
