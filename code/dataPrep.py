import torch.nn as nn
import pandas as pd
import numpy as np
import bottleneck as bn
from scipy import sparse
from sklearn import preprocessing
import sys

DATASET = sys.argv[1]
# order = ['iid','uid','score','bidtime', 'bidderrate', 'openbid', 'price', 'item','auction_type', 'winner']

def split_train_test_proportion(data, test_size=0.7):
    unique_iid = data['iid'].drop_duplicates().tolist()
    unique_uid = data['uid'].drop_duplicates().tolist()
    
    unique_iid = np.array(unique_iid)# array
    te_items = unique_iid[:-(int(unique_iid.size* test_size))]

    train_bid, train_buy, test_bid, test_buy = list(), list(), list(), list()
    posUsers, negUsers,testUser = dict(), dict(), dict()
    
    data_grouped_by_item = data.groupby('iid')
    for _, group in data_grouped_by_item:
        cur_iid = group.head(1)['iid'].tolist()[0]
        n_items_u = len(group)
        buyidx = -1
        if group['winner'].max() == 1:
            bids = group['score'].tolist()
            buyidx = bids.index(max(bids))
        
        if n_items_u >= 2:
            idx = np.zeros(n_items_u, dtype='bool')
            if buyidx >= 0:
                idx[buyidx] = True

            if cur_iid in te_items:
                test_bid.append(group[np.logical_not(idx)])
                test_buy.append(group[idx])

                if buyidx >= 0:
                    testUser[cur_iid] = group[idx]['uid'].tolist()[0]
            else: 
                idx = np.zeros(n_items_u, dtype='bool')
                if buyidx >= 0:
                    idx[buyidx] = True

                train_bid.append(group[np.logical_not(idx)])
                train_buy.append(group[idx])

        posUsers[cur_iid] = group['uid'].tolist()   
    train_bid = pd.concat(train_bid)
    train_buy = pd.concat(train_buy)
    test_bid = pd.concat(test_bid)
    test_buy = pd.concat(test_buy)

    for iid in posUsers:
        testUid = -1
        for i in range(len(unique_uid)):
            if i not in posUsers[iid] and i != testUid:
                if iid not in negUsers:
                    negUsers[iid]=list()
                negUsers[iid].append(i)
    return train_bid, train_buy, test_bid, test_buy, negUsers

def read_swp():
    data = pd.read_csv('Data/swp/auction_n5.csv', sep=';', engine='python')
    data = data[['auction_id','bid_user','bid_ct','bid_cp','bid_number']]

    aidList = data['auction_id'].drop_duplicates().tolist()
    df_meta=pd.read_csv("./Data/outcomes.tsv", header=0,sep='\t').dropna(how='any')
    df_meta = df_meta[df_meta['auction_id'].isin(aidList)]
    df_meta = df_meta[['auction_id','item','desc','retail','winner','price']]

    data=data.sort_values(by=['auction_id','bid_number'],ascending=False)
    data = data.drop_duplicates(['auction_id','bid_user'])

    data=data.sort_values(by=['auction_id','bid_number'],ascending=False)
    playcount_groupbyid = data[['bid_user']].groupby('bid_user', as_index=False)
    usercount = playcount_groupbyid.size()
    userList = usercount[usercount['size'] >= 2]['bid_user'].values 
    data = data[data['bid_user'].isin(userList)] 

    data = pd.merge(data, df_meta, how='left', on=['auction_id'])
    data['winner'] = data.apply(lambda x: 1 if x.bid_user==x.winner else 0,axis=1)
    
    data.columns=['iid','uid','bidtime','score','order','item','auction_type','openbid','winner','price']
    ItemEncoder = preprocessing.LabelEncoder()
    UserEncoder = preprocessing.LabelEncoder()
    data.iid = ItemEncoder.fit_transform(data.iid)
    data.uid = UserEncoder.fit_transform(data.uid)
    data.iid = data.iid.astype(int)
    data.uid = data.uid.astype(int)
    data.score = data.score.astype(float)
    print('item #', data['iid'].drop_duplicates().shape[0])
    print('user #',data['uid'].drop_duplicates().shape[0])
    return data

def read_ebay():
    data=pd.read_csv("Data/ebay/auction.csv", sep=',', engine='python')
    data.bidder=data.bidder.astype(str)
    data.columns=['iid','score','bidtime','uid','bidderrate','openbid','price','item','auction_type']

    data=data.sort_values(by=['iid','score'],ascending=False)
    data = data.drop_duplicates(['iid','uid'])
    data=data.sort_values(by=['iid','score'],ascending=False)
    playcount_groupbyid = data[['uid']].groupby('uid', as_index=False)
    usercount = playcount_groupbyid.size()
    userList = usercount[usercount['size'] >= 2]['uid'].values 
    data = data[data['uid'].isin(userList)] 
    data['winner'] = data.apply(lambda x: 1 if x.score==x.price else 0,axis=1)

    ItemEncoder = preprocessing.LabelEncoder()
    UserEncoder = preprocessing.LabelEncoder()
    data.iid = ItemEncoder.fit_transform(data.iid)
    data.uid = UserEncoder.fit_transform(data.uid)
    data.iid = data.iid.astype(int)
    data.uid = data.uid.astype(int)
    data.score = data.score.astype(float)
    print('item #', data['iid'].drop_duplicates().shape[0])
    print('user #',data['uid'].drop_duplicates().shape[0])
    return data

def popular(data, testBUY, topk=10):
    playcount_groupbyid = data[['uid']].groupby('uid', as_index=False)
    usercount = playcount_groupbyid.size().sort_values(by=['size'],ascending=False) 
    popularList = usercount['uid'].tolist()
    popularK = popularList[:topk]
    Count = 0
    for index, row in testBUY.iterrows():
        uid= int(row['uid'])
        if uid in popularK:
            Count+=1
    print('Popular HR{} by hit{}/total{}:----- {:.4f}'.format( str(topk) ,Count, testBUY.shape[0], Count/testBUY.shape[0]))

def calculate_ndgc(data, testBUY, topk=10):
    playcount_groupbyid = data[['uid']].groupby('uid', as_index=False)
    usercount = playcount_groupbyid.size().sort_values(by=['size'],ascending=False) 
    popularList = usercount['uid'].tolist()
    popularK = popularList[:topk]
    set = testBUY['uid'].tolist()
    ndcg = NDCG(set, popularK)
    print('Popular NDCG{}: {:.4f}'.format( str(topk), ndcg/testBUY.shape[0] ))

def randomMetric( testBUY):
    topks=[10,20,50]
    if DATASET =='swp':
        randomlist = np.random.randint(100,size=5932)
    else:
        randomlist = np.random.randint(100,size=3388)
    Count = 0
    hrs=[]
    ndcgs=[]
    for topk in topks:
        for index, row in testBUY.iterrows():
            uid= int(row['uid'])
            if uid in randomlist[:topk]:
                Count+=1
        hrs.append(Count/testBUY.shape[0])
        set = testBUY['uid'].tolist()
        ndcg = NDCG(set, randomlist[:topk])
        ndcgs.append(ndcg/testBUY.shape[0])
    print('HR10/20/50: ', [round(hrs[i],4) for i in range(3)])
    print('NDCG10/20/50: ', [round(ndcgs[i],4) for i in range(3)])

def DCG(A, set):
    dcg=0
    for i in range(len(A)):
        r_i = 0
        if A[i] in set:
            r_i = 1
        dcg +=(2 ** r_i-1) / np.log2((i+1)+1)
    return dcg

def IDCG(A, set): 
    A_temp1 = []
    A_temp0 = []
    for a in A:
        if a in set: A_temp1.append(a)
        else: A_temp0.append(a) 
    A_temp1.extend(A_temp0)
    return DCG(A_temp1, set)

def NDCG(A, set):
    dcg = DCG(A, set)
    idcg = IDCG(A, set)
    if dcg == 0 or idcg ==0:
        ndcg = 0
    else: ndcg = dcg/idcg
    return ndcg

def main():
    order = ['iid','uid','score','bidtime', 'bidderrate', 'item','auction_type','openbid', 'price','winner']
    if DATASET == 'swp':
        data = read_swp()
        data = data[['iid','uid','score','bidtime', 'order', 'item','auction_type','openbid', 'price','winner']]
    else:
        data = read_ebay()
        data = data[order]
    itemGroup = data.drop_duplicates('item')['item'].tolist()

    labelList = []
    for i in range(len(itemGroup)):
        labels = itemGroup[i].split('-')
        labelList += labels
    count = pd.value_counts(labelList)
    trainBID, trainBUY, testBID, testBUY, negUsers = split_train_test_proportion(data,test_size=0.7)# 会过滤掉参与人<2的商品
    print('train/test BUY items #: ', trainBUY.shape, testBUY.shape)

    data.to_csv('Data/MFdata.csv', sep=';', encoding='utf-8',index=False)
    trainBID.to_csv('Data/MFtrainBID1.csv', sep=';',encoding='utf-8',index=False)
    testBID.to_csv('Data/MFtestBID2.csv', sep=';',encoding='utf-8',index=False)
    trainBUY.to_csv('Data/MFtrainBUY3.csv', sep=';',encoding='utf-8',index=False)
    testBUY.to_csv('Data/MFtestBUY4.csv', sep=';',encoding='utf-8',index=False)

    # print('Based on 1+2')
    # popular(pd.concat([trainBID, testBID]),testBUY,10)
    # popular(pd.concat([trainBID, testBID]),testBUY,20)
    # popular(pd.concat([trainBID, testBID]),testBUY,50)

    # calculate_ndgc(pd.concat([trainBID, testBID]),testBUY,10)
    # calculate_ndgc(pd.concat([trainBID, testBID]),testBUY,20)
    # calculate_ndgc(pd.concat([trainBID, testBID]),testBUY,50)

    # print('Based on 3')
    # popular(trainBUY,testBUY,10)
    # popular(trainBUY,testBUY,20)
    # popular(trainBUY,testBUY,50)

    # calculate_ndgc(trainBUY,testBUY,10)
    # calculate_ndgc(trainBUY,testBUY,20)
    # calculate_ndgc(trainBUY,testBUY,50)

    # print('Based on 1+2+3')
    # popular(pd.concat([trainBID, testBID,trainBUY]),testBUY,10)
    # popular(pd.concat([trainBID, testBID,trainBUY]),testBUY,20)
    # popular(pd.concat([trainBID, testBID,trainBUY]),testBUY,50)

    # calculate_ndgc(pd.concat([trainBID, testBID,trainBUY]),testBUY,10)
    # calculate_ndgc(pd.concat([trainBID, testBID,trainBUY]),testBUY,20)
    # calculate_ndgc(pd.concat([trainBID, testBID,trainBUY]),testBUY,50)

    # print('Random:')
    # randomMetric(testBUY)

if __name__ == '__main__':
    main()