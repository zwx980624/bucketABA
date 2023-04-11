from torch.utils.data import Dataset
import math
import random
import numpy as np
import config
import pdb
from tqdm import tqdm

class SubDataSet:
    def __init__(self, train_dataset=None, valid_dataset=None, test_dataset=None):
        
        self.train_dataset = [] if train_dataset == None else train_dataset
        self.valid_dataset = [] if valid_dataset == None else valid_dataset
        self.test_dataset = [] if test_dataset == None else test_dataset

class BasketData(Dataset):
    def __init__(self, input_dir, args):
        print("--------------Begin Data Process--------------")
        self.args = args
        self.user_basket_item_dict_old = self.get_dict(input_dir)
        self.max_basket_num = args.max_basket_num
        self.max_basket_size = args.max_basket_size
        self.user_basket_item_dict = dict()

        item_set = set()
        print("userid:", max(self.user_basket_item_dict_old.keys()), len(self.user_basket_item_dict_old))
        max_item_id = 0
        for userid in self.user_basket_item_dict_old.keys():
            seq = self.user_basket_item_dict_old[userid]
            for b_id, basket in enumerate(seq):
                for item in basket:
                    item_set.add(item)
                    if max_item_id < item:
                        max_item_id = item
        print("itemid:", max_item_id, len(item_set))
        # pdb.set_trace()
        self.item_num = max_item_id + 2
        self.user_num = max(self.user_basket_item_dict_old.keys()) + 2
        self.item_pad = max_item_id + 1

        # 先截断
        for userid in self.user_basket_item_dict_old.keys():
            seq = self.user_basket_item_dict_old[userid]
            if len(seq) > self.max_basket_num:
                seq = seq[-self.max_basket_num:]
            # if len(seq) < 1 + next_k: continue
            for b_id, basket in enumerate(seq):
                if len(basket) > self.max_basket_size:
                    seq[b_id] = basket[-self.max_basket_size:]
            self.user_basket_item_dict[userid] = seq
        
        
        # 划分训练测试集
        self.train_userid_list = list(self.user_basket_item_dict.keys())[:math.ceil(0.9 * len(list(self.user_basket_item_dict.keys())))]
        # CVR_data
        
        print("generating instance")
        self.CVR_dataset, self.CVABR_dataset, self.CVBCVAR_dataset, self.neg_sample_pool = self.create_instance()
        
        self.random_instance_balance()
        # pdb.set_trace()
        print('done')
    
    def __getitem__(self, idx):
        # if (idx == 0):
        #     pdb.set_trace()
        # if (idx >= self.__len__()):
        #     pdb.set_trace()
        ret = self.sampled_train_comb[(idx)*self.args.batch_size: (idx+1)*self.args.batch_size]
        # group一下
        U = [_[0] for _ in ret]
        S = [_[1] for _ in ret]
        A = [_[2] for _ in ret]
        B = [_[3] for _ in ret]
        L1 = [_[4] for _ in ret]
        L2 = [_[5] for _ in ret]
        L3 = [_[6] for _ in ret]
        if (len(U) == 0):
            pdb.set_trace()
        return (U, S, A, B, L1, L2, L3)

    def __len__(self):
        return (len(self.sampled_train_comb) // self.args.batch_size)


        
    def random_instance_balance(self):
        CVR_num = len(self.CVR_dataset.train_dataset)
        CVABR_num = len(self.CVABR_dataset.train_dataset)
        CVBCVAR_num = len(self.CVBCVAR_dataset.train_dataset)

        CVR_sample_rate = self.args.CVR_sample_rate
        CVABR_balance = self.args.CVABR_balance
        CVBCVAR_balance = self.args.CVBCVAR_balance

        new_CVR_num = int(CVR_num * CVR_sample_rate)
        new_CVABR_num = int(new_CVR_num * CVABR_balance)
        new_CVBCVAR_num = int(new_CVR_num * CVBCVAR_balance)


        self.sampled_CVR_train = random.choices(self.CVR_dataset.train_dataset, k=new_CVR_num)
        self.sampled_CVABR_train = random.choices(self.CVABR_dataset.train_dataset, k=new_CVABR_num)
        self.sampled_CVBCVAR_train = random.choices(self.CVBCVAR_dataset.train_dataset, k=new_CVBCVAR_num)

        # 合并, (U, S, A, (B), L1, L2, L3)
        self.sampled_CVR_train = [(_[0], _[1], _[2], _[2], _[3], -1, -1) for _ in self.sampled_CVR_train]
        self.sampled_CVABR_train = [(_[0], _[1], _[2], _[3], -1, _[4], -1) for _ in self.sampled_CVABR_train]
        self.sampled_CVBCVAR_train = [(_[0], _[1], _[2], _[3], -1, -1, _[4]) for _ in self.sampled_CVBCVAR_train]

        self.sampled_train_comb = self.sampled_CVR_train + self.sampled_CVABR_train + self.sampled_CVBCVAR_train

        random.shuffle(self.sampled_train_comb)



    def create_instance(self):

        CVR_dataset = SubDataSet()
        CVABR_dataset = SubDataSet()
        CVBCVAR_dataset = SubDataSet()
        neg_sample_pool = {}

        neg_ratio = self.args.cvr_neg_ratio

        # 剩余10% 是valid集
        itemnum = 0
        for userid in self.user_basket_item_dict.keys():
            seq = self.user_basket_item_dict[userid]
            for basket in seq:
                for item in basket:
                    if item > itemnum:
                        itemnum = item
        self.itemnum = itemnum + 1
        self.item_list = [i for i in range(0, itemnum)]

        for userid in self.user_basket_item_dict.keys():
            if userid in self.train_userid_list:
                seq = self.user_basket_item_dict[userid][:-1]
            # 验证集不需要负采样
            seq_pool = []
            for basket in seq:
                seq_pool = seq_pool + basket
            neg_sample_pool[userid] = list(set(self.item_list) - set(seq_pool))
        
        for userid in tqdm(self.user_basket_item_dict.keys()):
            if userid in self.train_userid_list: # train and test
                seq = self.user_basket_item_dict[userid]
                if (len(seq) < 3): # at least: history_sess*1+train_sess+test_sess
                    continue
                history = []
                train_seq = seq[:-1] # leave-one-out
                for basketid, basket in enumerate(train_seq):
                    if len(basket) > self.max_basket_size:
                        basket = basket[-self.max_basket_size:]
                    else:
                        padd_num = self.max_basket_size - len(basket)
                        padding_item = [self.item_pad] * padd_num
                        basket = basket + padding_item
                    history.append(basket)
                    if len(history) == 1: continue
                    U = userid  
                    S = history[:-1]
                    tgt_basket = train_seq[basketid]
                    # CVR 
                    for item in tgt_basket:
                        T = item
                        N = random.sample(neg_sample_pool[userid], neg_ratio)
                        CVR_dataset.train_dataset.append((U, S, T, 1))
                        for n in N:
                            CVR_dataset.train_dataset.append((U, S, n, 0))
                    # CVABR
                    for item1 in tgt_basket:
                        N1 = random.sample(neg_sample_pool[userid], neg_ratio)
                        for item2 in tgt_basket:
                            N2 = random.sample(neg_sample_pool[userid], neg_ratio)
                            if item1 == item2:
                                continue
                            CVABR_dataset.train_dataset.append((U, S, item1, item2, 1)) # True
                            CVABR_dataset.train_dataset.append((U, S, item1, N2[0], 0)) # False
                            CVABR_dataset.train_dataset.append((U, S, N1[0], item2, 0)) # False
                            CVABR_dataset.train_dataset.append((U, S, N1[0], N2[0], 0)) # False
                    # CVBCVAR
                    for item1 in tgt_basket:
                        N1 = random.sample(neg_sample_pool[userid], neg_ratio)
                        for item2 in tgt_basket:
                            if item1 == item2:
                                continue
                            CVBCVAR_dataset.train_dataset.append((U, S, item1, item2, 1)) # True
                            for n1 in N1:
                                CVBCVAR_dataset.train_dataset.append((U, S, n1, item2, 0)) # False
                            
                test_seq = seq
                history = []
                for basketid, basket in enumerate(test_seq):
                    if len(basket) > self.max_basket_size:
                        basket = basket[-self.max_basket_size:]
                    else:
                        padd_num = self.max_basket_size - len(basket)
                        padding_item = [self.item_pad] * padd_num
                        basket = basket + padding_item
                    history.append(basket)
                U = userid
                S = list(history[:-1])
                T_basket = history[-1]
                CVR_dataset.test_dataset.append((U, S, T_basket, 1))
                CVABR_dataset.test_dataset.append((U, S, T_basket, 1))
                CVBCVAR_dataset.test_dataset.append((U, S, T_basket, 1))


            else: # valid
                seq = self.user_basket_item_dict[userid]
                if (len(seq) < 2): # at least: history_sess*1+valid_sess
                    continue
                history = []
                valid_seq = seq
                for basketid, basket in enumerate(valid_seq):
                    if len(basket) > self.max_basket_size:
                        basket = basket[-self.max_basket_size:]
                    else:
                        padd_num = self.max_basket_size - len(basket)
                        padding_item = [self.item_pad] * padd_num
                        basket = basket + padding_item
                    history.append(basket)
                    if len(history) == 1: continue
                    if len(history) < len(valid_seq): continue
                    U = userid  
                    S = history[:-1] 
                    T_basket = history[-1]
                    CVR_dataset.valid_dataset.append((U, S, T_basket, 1))
                    CVABR_dataset.valid_dataset.append((U, S, T_basket, 1))
                    CVBCVAR_dataset.valid_dataset.append((U, S, T_basket, 1))

        return CVR_dataset, CVABR_dataset, CVBCVAR_dataset, neg_sample_pool
    
    def get_dict(self, path):
        f = open(path, 'r')
        a = f.read()
        geted_dict = eval(a)
        f.close()
        return geted_dict


if __name__ == "__main__":
    args = config.args
    ds = BasketData("./my_data/Dunn/user_date_tran_dict_new_small.txt", args)
    print(len(ds))
    for i in range(len(ds)):
        item = ds[i]
        print(i)
        # pdb.set_trace()
        # print("ok")
    # a = SubDataSet()
    # b = SubDataSet()
    # a.train_dataset.append(1)
    # print(a.train_dataset)
    # print(b.train_dataset)