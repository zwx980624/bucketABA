import numpy as np
import random
import sys
import csv
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pdb
import config
from datasets import BasketData
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

num_iter = 10

past_chunk = 0
future_chunk = 1
hidden_size = 32
num_layers = 1

# only one can be set 1
use_embedding = 1
use_linear_reduction = 0
###
atten_decoder = 1
use_dropout = 0
use_average_embedding = 1

weight = 10
labmda = 0
topk_labels = 3

# It should be the same as the reductioned input in decoder's cat function

teacher_forcing_ratio = 0
MAX_LENGTH = 1000
learning_rate = 0.001
optimizer_option = 2
print_val = 3000
use_cuda = torch.cuda.is_available()


class EncoderRNN_new(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN_new, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduction = nn.Linear(input_size, hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.time_embedding = nn.Embedding(input_size, hidden_size)
        self.time_weight = nn.Linear(input_size, input_size)
        if use_embedding or use_linear_reduction:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            self.gru = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, input, hidden):
        if use_embedding:
            list = Variable(torch.LongTensor(input).view(-1, 1))
            if use_cuda:
                list = list.cuda()
            average_embedding = Variable(torch.zeros(hidden_size)).view(1, 1, -1)
            # sum_embedding = Variable(torch.zeros(hidden_size)).view(1,1,-1)
            vectorized_input = Variable(torch.zeros(self.input_size)).view(-1)
            if use_cuda:
                average_embedding = average_embedding.cuda()
                # sum_embedding = sum_embedding.cuda()
                vectorized_input = vectorized_input.cuda()

            for ele in list:
                embedded = self.embedding(ele).view(1, 1, -1)
                tmp = average_embedding.clone()
                average_embedding = tmp + embedded
                # embedded = self.time_embedding(ele).view(1, 1, -1)
                # tmp = sum_embedding.clone()
                # sum_embedding = tmp + embedded
                vectorized_input[ele] = 1

            # normalize_length = Variable(torch.LongTensor(len(idx_list)))
            # if use_cuda:
            #     normalize_length = normalize_length.cuda()
            if use_average_embedding:
                tmp = [1] * hidden_size
                length = Variable(torch.FloatTensor(tmp))
                if use_cuda:
                    length = length.cuda()
                # for idx in range(hidden_size):
                real_ave = average_embedding.view(-1) / length
                average_embedding = real_ave.view(1, 1, -1)

            embedding = average_embedding
        else:
            tensorized_input = torch.from_numpy(input).clone().type(torch.FloatTensor)
            inputs = Variable(torch.unsqueeze(tensorized_input, 0).view(1, -1))
            if use_cuda:
                inputs = inputs.cuda()
            if use_linear_reduction == 1:
                reduced_input = self.reduction(inputs)
            else:
                reduced_input = inputs

            embedding = torch.unsqueeze(reduced_input, 0)

        output, hidden = self.gru(embedding, hidden) # here encode
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(num_layers, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


#


class AttnDecoderRNN_new(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p=0.2, max_length=MAX_LENGTH):
        super(AttnDecoderRNN_new, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_embedding or use_linear_reduction:
            self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
            self.attn1 = nn.Linear(self.hidden_size + output_size, self.hidden_size)
        else:
            self.attn = nn.Linear(self.hidden_size + self.output_size, self.output_size)

        if use_embedding or use_linear_reduction:
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.attn_combine3 = nn.Linear(self.hidden_size * 2 + output_size, self.hidden_size)
        else:
            self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)
        self.attn_combine5 = nn.Linear(self.output_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.reduction = nn.Linear(self.output_size, self.hidden_size)
        if use_embedding or use_linear_reduction:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, history_record, last_hidden):
        # pdb.set_trace()
        if use_embedding:
            list = Variable(torch.LongTensor(input).view(-1, 1))
            if use_cuda:
                list = list.cuda()
            average_embedding = Variable(torch.zeros(hidden_size)).view(1, 1, -1)
            if use_cuda:
                average_embedding = average_embedding.cuda()

            for ele in list:
                embedded = self.embedding(ele).view(1, 1, -1)
                tmp = average_embedding.clone()
                average_embedding = tmp + embedded

            if use_average_embedding:
                tmp = [1] * hidden_size
                length = Variable(torch.FloatTensor(tmp))
                if use_cuda:
                    length = length.cuda()
                # for idx in range(hidden_size):
                real_ave = average_embedding.view(-1) / length
                average_embedding = real_ave.view(1, 1, -1)

            embedding = average_embedding
        else:
            tensorized_input = torch.from_numpy(input).clone().type(torch.FloatTensor)
            inputs = Variable(torch.unsqueeze(tensorized_input, 0).view(1, -1))
            if use_cuda:
                inputs = inputs.cuda()
            if use_linear_reduction == 1:
                reduced_input = self.reduction(inputs)
            else:
                reduced_input = inputs

            embedding = torch.unsqueeze(reduced_input, 0)

        if use_dropout:
            droped_ave_embedded = self.dropout(embedding)
        else:
            droped_ave_embedded = embedding

        history_context = Variable(torch.FloatTensor(history_record).view(1, -1))
        if use_cuda:
            history_context = history_context.cuda()

        attn_weights = F.softmax(
            self.attn(torch.cat((droped_ave_embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        element_attn_weights = F.softmax(
            self.attn1(torch.cat((history_context, hidden[0]), 1)), dim=1)

        # attn_applied = torch.bmm(element_attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        # attn_embedd = element_attn_weights * droped_ave_embedded[0]

        output = torch.cat((droped_ave_embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        # output = torch.cat((droped_ave_embedded[0], attn_applied[0], time_coef.unsqueeze(0)), 1)
        # output = self.attn_combine3(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        linear_output = self.out(output[0]) # 1 * 5002
        # output_user_item = F.softmax(linear_output)

        value = torch.sigmoid(self.attn_combine5(history_context).unsqueeze(0))

        one_vec = Variable(torch.ones(self.output_size).view(1, -1))
        if use_cuda:
            one_vec = one_vec.cuda()

        # ones_set = torch.index_select(value[0,0], 1, ones_idx_set[:, 1])
        res = history_context.clone()
        res[history_context != 0] = 1

        output = F.softmax(linear_output * (one_vec - res * value[0]) + history_context * value[0], dim=1)

        return output.view(1, -1), hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(num_layers, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class custom_MultiLabelLoss_torch(nn.modules.loss._Loss):
    def __init__(self):
        super(custom_MultiLabelLoss_torch, self).__init__()

    def forward(self, pred, target, weights):
        mseloss = torch.sum(weights * torch.pow((pred - target), 2))
        pred = pred.data
        target = target.data
        #
        ones_idx_set = (target == 1).nonzero()
        zeros_idx_set = (target == 0).nonzero()
        # zeros_idx_set = (target == -1).nonzero()
        
        ones_set = torch.index_select(pred, 1, ones_idx_set[:, 1])
        zeros_set = torch.index_select(pred, 1, zeros_idx_set[:, 1])
        
        repeat_ones = ones_set.repeat(1, zeros_set.shape[1])
        repeat_zeros_set = torch.transpose(zeros_set.repeat(ones_set.shape[1], 1), 0, 1).clone()
        repeat_zeros = repeat_zeros_set.reshape(1, -1)
        difference_val = -(repeat_ones - repeat_zeros)
        exp_val = torch.exp(difference_val)
        exp_loss = torch.sum(exp_val)
        normalized_loss = exp_loss / (zeros_set.shape[1] * ones_set.shape[1])
        set_loss = Variable(torch.FloatTensor([labmda * normalized_loss]), requires_grad=True)
        if use_cuda:
            set_loss = set_loss.cuda()
        loss = mseloss + set_loss
        #loss = mseloss
        return loss

class BasketTrans(nn.Module):
    def __init__(self, item_embedding):
        super(BasketTrans, self).__init__()
    
        self.item_embedding = item_embedding

    def forward(self, S):
        # S: B * BN(?) * BS
        S = [s[-1] for s in S] # S: B * BS
        S = torch.tensor(S, dtype=torch.long)
        if use_cuda:
            S = S.cuda()
        ret = self.item_embedding(S) # B * BS * E
        return torch.sum(ret, dim=1) # B * E

        

class BasketABAModel(nn.Module):
    def __init__(self, item_num, user_num, args):
        super(BasketABAModel, self).__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.item_embedding = nn.Embedding(self.item_num, args.hidden_size)
        self.usr_embedding = nn.Embedding(self.user_num, args.hidden_size)
        self.encoder = BasketTrans(self.item_embedding)


    def forward(self, item):
        U, S, A, B, L1, L2, L3 = item
        # pdb.set_trace()
        
        U = torch.tensor(U, dtype=torch.long)
        A = torch.tensor(A, dtype=torch.long)
        B = torch.tensor(B, dtype=torch.long)
        
        if use_cuda:
            U = U.cuda()
            A = A.cuda()
            B = B.cuda()
        seq_emb = self.encoder(S)
        usr_emb = self.usr_embedding(U)
        
        itemA_emb = self.item_embedding(A)
        itemB_emb = self.item_embedding(B)
        # pdb.set_trace()
        try:
            logit = torch.matmul((usr_emb + seq_emb), itemA_emb.T)
        except Exception as e:
            pdb.set_trace()
            print("here")
        # pred = torch.sigmoid(logit) # B * 1
        return logit
    
    def eval_all(self, item):
        U, S, A_busk, _ = item
        U = torch.tensor([U], dtype=torch.long)
        if use_cuda:
            U = U.cuda()
        seq_emb = self.encoder([S]) # 1 * E
        usr_emb = self.usr_embedding(U) # 1 * E
        
        all_item = self.item_embedding.weight # N * E

        logit = torch.matmul((usr_emb + seq_emb), all_item.T) # 1 * N
        pred = torch.sigmoid(logit) # 1 * N

        return pred

class BucketABALoss(nn.modules.loss._Loss):
    def __init__(self):
        super(BucketABALoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, logit, item):
        # pdb.set_trace()
        target = [0 if x == -1 else x for x in item[-3]]
        target = torch.tensor(target, dtype=torch.long)
        if use_cuda:
            target = target.cuda()
        return self.loss(logit, target)


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(data, model, criterion, args):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-11, weight_decay=0)

    total_iter = 0
    for epoch in range(args.n_iters):

        for j in tqdm(range(len(data))):
            # (U, S, A, B, L1, L2, L3)
            item = data[j]
            optimizer.zero_grad()
            logit = model(item)
            loss = criterion(logit, item)

            loss.backward()
            optimizer.step()

            print_loss_total += loss.data.cpu().detach()
            plot_loss_total += loss.data.cpu().detach()

            total_iter += 1

        print_loss_avg = print_loss_total / len(data)
        print_loss_total = 0
        print('%s (%d %d%%) %.6f' % (timeSince(start, total_iter / (args.n_iters * len(data))), total_iter, total_iter / (args.n_iters * len(data)) * 100, print_loss_avg))

        recall, ndcg, hr = evaluate(data, model, "valid", 20)
        print(f"valid result: recall={recall}, ndcg={ndcg}, hr={hr}")

        filepath = './models/' + (args.model_version) + '_model_epoch' + str(int(epoch))
        torch.save(model, filepath)
        # filepath = './models/decoder' + (args.model_version) + '_model_epoch' + str(int(epoch))
        # torch.save(decoder, filepath)
        print('Finish epoch: ' + str(epoch))
        print('Model is saved.')
        sys.stdout.flush()
    # showPlot(plot_losses)
    # print('The loss: ' + str(print_loss_total))


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

cosine_sim = []
pair_cosine_sim = []


def decoding_next_k_step(encoder, decoder, input_variable, target_variable, output_size, k, activate_codes_num):
    encoder_hidden = encoder.initHidden()

    input_length = len(input_variable)
    encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size))
    if use_cuda:
        encoder_outputs = encoder_outputs.cuda()

    loss = 0

    history_record = np.zeros(output_size)
    count = 0
    for ei in range(input_length - 1):
        if ei == 0:
            continue
        for ele in input_variable[ei]:
            history_record[ele] += 1
        count += 1

    history_record = history_record / count

    for ei in range(input_length - 1):
        if ei == 0:
            continue
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei - 1] = encoder_output[0][0]

        for ii in range(k):
            vectorized_target = np.zeros(output_size)
            for idx in target_variable[ii + 1]:
                vectorized_target[idx] = 1

            vectorized_input = np.zeros(output_size)
            for idx in input_variable[ei]:
                vectorized_input[idx] = 1

    decoder_input = input_variable[input_length - 2]

    decoder_hidden = encoder_hidden
    last_hidden = decoder_hidden
    # Without teacher forcing: use its own predictions as the next input
    num_str = 0
    topk = 400
    decoded_vectors = []
    prob_vectors = []
    cout = 0
    for di in range(k):
        if atten_decoder:
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, history_record, last_hidden)
        else:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(topk)
        ni = topi[0][0]

        vectorized_target = np.zeros(output_size)
        for idx in target_variable[di + 1]:
            vectorized_target[idx] = 1

        # target_topi = vectorized_target.argsort()[::-1][:topk]
        # activation_bound

        count = 0
        start_idx = -1
        end_idx = output_size
        if activate_codes_num > 0:
            pick_num = activate_codes_num
        else:
            pick_num = np.sum(vectorized_target)
            # print(pick_num)

        tmp = []
        for ele in range(len(topi[0])):
            if count >= pick_num:
                break
            tmp.append(topi[0][ele])
            count += 1

        decoded_vectors.append(tmp)
        decoder_input = tmp
        tmp = []
        for i in range(topk):
            tmp.append(topi[0][i])
        prob_vectors.append(tmp)

    return decoded_vectors, prob_vectors


import bottleneck as bn


def top_n_indexes(arr, n):
    idx = bn.argpartition(arr, arr.size - n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


def get_precision_recall_Fscore(groundtruth, pred):
    a = groundtruth
    b = pred
    correct = 0
    truth = 0
    positive = 0

    for idx in range(len(a)):
        if a[idx] == 1:
            truth += 1
            if b[idx] == 1:
                correct += 1
        if b[idx] == 1:
            positive += 1

    flag = 0
    if 0 == positive:
        precision = 0
        flag = 1
        # print('postivie is 0')
    else:
        precision = correct / positive
    if 0 == truth:
        recall = 0
        flag = 1
        # print('recall is 0')
    else:
        recall = correct / truth

    if flag == 0 and precision + recall > 0:
        F = 2 * precision * recall / (precision + recall)
    else:
        F = 0
    return precision, recall, F, correct


def get_F_score(prediction, test_Y):
    jaccard_similarity = []
    prec = []
    rec = []

    count = 0
    for idx in range(len(test_Y)):
        pred = prediction[idx]
        T = 0
        P = 0
        correct = 0
        for id in range(len(pred)):
            if test_Y[idx][id] == 1:
                T = T + 1
                if pred[id] == 1:
                    correct = correct + 1
            if pred[id] == 1:
                P = P + 1

        if P == 0 or T == 0:
            continue
        precision = correct / P
        recall = correct / T
        prec.append(precision)
        rec.append(recall)
        if correct == 0:
            jaccard_similarity.append(0)
        else:
            jaccard_similarity.append(2 * precision * recall / (precision + recall))
        count = count + 1

    print(
        'average precision: ' + str(np.mean(prec)))
    print('average recall : ' + str(
        np.mean(rec)))
    print('average F score: ' + str(
        np.mean(jaccard_similarity)))


def get_DCG(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1

    return dcg


def get_NDCG(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1
    idcg = 0
    num_real_item = np.sum(groundtruth)
    num_item = int(min(num_real_item, k))
    for i in range(num_item):
        idcg += (1) / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg


def get_HT(groundtruth, pred_rank_list, k):
    count = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            return 1
        count += 1

    return 0


# def evaluate(data_chunk, encoder, decoder, output_size, test_key_set, next_k_step, activate_codes_num):
def evaluate(data, model, mode, activate_codes_num):
    prec = []
    rec = []
    F = []
    prec1 = []
    rec1 = []
    F1 = []
    prec2 = []
    rec2 = []
    F2 = []
    prec3 = []
    rec3 = []
    F3 = []
    length = np.zeros(3)
    

    NDCG = []
    n_hit = 0
    count = 0
    next_k_step = 1
    TOPK = activate_codes_num

    if mode == "test":
        dataset = data.CVR_dataset.test_dataset
    elif mode == "valid":
        dataset = data.CVR_dataset.valid_dataset
    else:
        print("error evaluate mode: " + mode)
        return

    for iter in tqdm(range(len(dataset))):
        item = dataset[iter] # (U, S, A_busk, L)
        targets = item[2]

        pred_all = model.eval_all(item) # 1(next_k) * item_num(or random select 100)

        topv, sorted_item = pred_all.data.cpu().detach().topk(TOPK)

        # output_vectors, prob_vectors = decoding_next_k_step(encoder, decoder, input_variable, target_variable,
        #                                                     output_size, next_k_step, activate_codes_num)

        hit = 0
        for idx in range(len(sorted_item)):
            # for idx in [2]:
            vectorized_target = np.zeros(data.item_num)
            for ii in targets:
                vectorized_target[ii] = 1

            vectorized_output = np.zeros(data.item_num)
            for ii in sorted_item[idx]:
                vectorized_output[ii] = 1

            precision, recall, Fscore, correct = get_precision_recall_Fscore(vectorized_target, vectorized_output)
            prec.append(precision)
            rec.append(recall)
            F.append(Fscore)
            if idx == 0:
                prec1.append(precision)
                rec1.append(recall)
                F1.append(Fscore)
            elif idx == 1:
                prec2.append(precision)
                rec2.append(recall)
                F2.append(Fscore)
            elif idx == 2:
                prec3.append(precision)
                rec3.append(recall)
                F3.append(Fscore)
            # length[idx] += np.sum(target_variable[1 + idx])
            # target_topi = prob_vectors[idx]
            # pdb.set_trace()
            hit += get_HT(vectorized_target, sorted_item[idx], activate_codes_num)
            ndcg = get_NDCG(vectorized_target, sorted_item[idx], activate_codes_num)
            NDCG.append(ndcg)
        if hit == next_k_step:
            n_hit += 1

    # print('average precision of subsequent sets' + ': ' + str(np.mean(prec)) + ' with std: ' + str(np.std(prec)))
    # print('average recall' + ': ' + str(np.mean(rec)) + ' with std: ' + str(np.std(rec)))
    # print('average F score of subsequent sets' + ': ' + str(np.mean(F)) + ' with std: ' + str(np.std(F)))
    # print('average precision of 1st' + ': ' + str(np.mean(prec1)) + ' with std: ' + str(np.std(prec1)))
    # print('average recall of 1st' + ': ' + str(np.mean(rec1)) + ' with std: ' + str(np.std(rec1)))
    # print('average F score of 1st' + ': ' + str(np.mean(F1)) + ' with std: ' + str(np.std(F1)))
    # print('average precision of 2nd' + ': ' + str(np.mean(prec2)) + ' with std: ' + str(np.std(prec2)))
    # print('average recall of 2nd' + ': ' + str(np.mean(rec2)) + ' with std: ' + str(np.std(rec2)))
    # print('average F score of 2nd' + ': ' + str(np.mean(F2)) + ' with std: ' + str(np.std(F2)))
    # print('average precision of 3rd' + ': ' + str(np.mean(prec3)) + ' with std: ' + str(np.std(prec3)))
    # print('average recall of 3rd' + ': ' + str(np.mean(rec3)) + ' with std: ' + str(np.std(rec3)))
    # print('average F score of 3rd' + ': ' + str(np.mean(F3)) + ' with std: ' + str(np.std(F3)))
    # print('average NDCG: ' + str(np.mean(NDCG)))
    # print('average hit rate: ' + str(n_hit / len(test_key_set)))
    return np.mean(rec), np.mean(NDCG), n_hit / len(dataset)

def partition_the_data(data_chunk, key_set, next_k_step):
    filtered_key_set = []
    for key in key_set:
        if len(data_chunk[past_chunk][key]) <= 3:
            continue
        if len(data_chunk[future_chunk][key]) < 2 + next_k_step:
            continue
        filtered_key_set.append(key)

    training_key_set = filtered_key_set[0:int(4 / 5 * len(filtered_key_set))]
    print('Number of training instances: ' + str(len(training_key_set)))
    test_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)):]
    return training_key_set, test_key_set

def partition_the_data_validate(data_chunk, key_set, next_k_step):
    filtered_key_set = []
    for key in key_set:
        if len(data_chunk[past_chunk][key]) <= 3:
            continue
        if len(data_chunk[future_chunk][key]) < 2 + next_k_step:
            continue
        filtered_key_set.append(key) # 按用户划分训练测试集

    training_key_set = filtered_key_set[0:int(4 / 5 * len(filtered_key_set)*0.9)]
    validation_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)*0.9):int(4 / 5 * len(filtered_key_set))]
    print('Number of training instances: ' + str(len(training_key_set)))
    test_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)):]
    return training_key_set, validation_key_set, test_key_set

def get_codes_frequency_no_vector(X, num_dim, key_set):
    result_vector = np.zeros(num_dim)
    for pid in key_set:
        for idx in X[pid]:
            result_vector[idx] += 1
    return result_vector


# The first two parameters are the past records and future records, respectively.
# The main function consists of two mode which is decisded by the argv[5]. If training is 1, it is training mode. If
# training is 0, it is test mode. model_version is the name of the model. next_k_step is the number of steps we predict.
# model_epoch is the model generated by the model_epoch-th epoch.
def main(argv):
    args = config.args

    data = BasketData(args.data_path, args)

    model_version = args.model_version

    directory = './models/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    model = BasketABAModel(data.item_num, data.user_num, args)
    # encoder = BucketEncoderTrans(item_num, hidden_size, num_layers)
    # attn_decoder1 = AttnDecoderRNN_new(hidden_size, item_num, num_layers, dropout_p=0.1)
    criterion = BucketABALoss()

    if use_cuda:
        model = model.cuda()
        criterion.cuda()

    if not args.only_test:
        trainIters(data, model, criterion, args)

    # test
    for i in [20, 40]:
        valid_recall = []
        valid_ndcg = []
        valid_hr = []
        recall_list = []
        ndcg_list = []
        hr_list = []
        print('k = ' + str(i))
        for model_epoch in range(args.n_iters):
            print('Epoch: ', model_epoch)
            model_pathes = './models/' + str(model_version) + '_model_epoch' + str(model_epoch)

            model_instance = torch.load(model_pathes)

            recall, ndcg, hr = evaluate(data, model_instance, "test", i)
            print(f"test recall={recall}, ndcg={ndcg}, hr={hr}")
            valid_recall.append(recall)
            valid_ndcg.append(ndcg)
            valid_hr.append(hr)
            recall, ndcg, hr = evaluate(data, model_instance, "valid", i)
            print(f"valid recall={recall}, ndcg={ndcg}, hr={hr}")
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            hr_list.append(hr)
        valid_recall = np.asarray(valid_recall)
        valid_ndcg = np.asarray(valid_ndcg)
        valid_hr = np.asarray(valid_hr)
        idx1 = valid_recall.argsort()[::-1][0]
        idx2 = valid_ndcg.argsort()[::-1][0]
        idx3 = valid_hr.argsort()[::-1][0]
        print('max valid recall results:')
        print('Epoch: ', idx1)
        print('recall: ', recall_list[idx1])
        print('ndcg: ', ndcg_list[idx1])
        print('phr: ', hr_list[idx1])

        print('max valid ndcg results:')
        print('Epoch: ', idx2)
        print('recall: ', recall_list[idx2])
        print('ndcg: ', ndcg_list[idx2])
        print('phr: ', hr_list[idx2])

        print('max valid phr results:')
        print('Epoch: ', idx3)
        print('recall: ', recall_list[idx3])
        print('ndcg: ', ndcg_list[idx3])
        print('phr: ', hr_list[idx3])

if __name__ == '__main__':
    main(sys.argv)
