import time
import torch
import numpy as np
from train_eval import train, use, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

Use_Open = False

with open('THUCNews/data/use.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 去除每行末尾的换行符
lines = [line.strip() for line in lines]

# 在每行末尾添加制表符和0
for i, line in enumerate(lines):
    lines[i] = line + '\t0'

# 写入新文件
with open('THUCNews/data/use_1.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  #TextCNN, TextRNN,
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data, use_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    use_iter = build_iterator(use_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)

    if Use_Open == False:
        train(config, model, train_iter, dev_iter, test_iter,writer)
    else:
        #use
        use(config, model, use_iter, writer)
