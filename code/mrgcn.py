import os 
import numpy as np
import random
import sys
import pandas as pd
from collections import deque
import random
import math
import datetime
import torch
import dataloader as dataloader_ebay
import models
import metrics
import time

import warnings
warnings.filterwarnings("ignore") 

dataset=sys.argv[1]
UseSalePrice=int(sys.argv[2].split("=")[1])
UseBidding=int(sys.argv[3].split("=")[1])
UserCL=int(sys.argv[4].split("=")[1])
alpha=float(sys.argv[5].split("=")[1])
beta=float(sys.argv[6].split("=")[1])
gamma=int(sys.argv[7].split("=")[1])
K=int(sys.argv[8].split("=")[1])
emb = int(sys.argv[9].split("=")[1])

LEARNRATE=0.00003 
TIP='a'+str(alpha)+'_b'+str(beta)+'_G_U_'+str(gamma)+'V_emb'+str(emb)+'_K'+str(K)
FILELABEL = TIP+'_seed_'
EMBEDDING_SIZE= emb # emb
BATCH_SIZE = emb   # emb
EPOCH_MAX = 2000
NEGSAMPLES=1
TOPK=[10,20,50]
UseItemData=1
UseGraphData=0


def to_np(tensor):
  return tensor.detach().numpy()

def train(loader, train,seed,ItemData=False,lr=0.00002):
    ITEMDATA,labelList=loader._get_ItemData()
    labelList = [f"An auction item of the {label}" for label in labelList]
    UserFeatures=np.identity(USER_NUM)
    # ItemFeatures=np.identity(ITEM_NUM)
    ItemFeatures=[]

    if(ItemData):
      ItemFeatures=ITEMDATA
      print(ItemFeatures.shape)     

    print(UserFeatures.shape)
    print(ItemFeatures.shape)
    samples_per_batch = len(train) // int(BATCH_SIZE/(NEGSAMPLES+1)) 
    w_user = torch.from_numpy(UserFeatures)
    w_item = torch.from_numpy(ItemFeatures)
    del UserFeatures 
    del ItemFeatures

    adj_mat_buy,adj_mat_bid = loader._generate_Graph(dictUsers)
    model = models.DCL(USER_NUM, ITEM_NUM, adj_mat_buy,adj_mat_bid, w_user, w_item,UseSalePrice,dataset,labelList,alpha,beta,EMBEDDING_SIZE,K)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    model = model.float().to(DEVICE)

    print("epoch, train_err,train_saleerror,HitRatio10,HitRatio20,HitRatio50,NDCG10,NDCG20,NDCG50")
    errors = deque(maxlen=samples_per_batch)
    saleerrors = deque(maxlen=samples_per_batch)
    now = datetime.datetime.now()
    textTrain_file = open("./Output"+dataset+"/"+FILELABEL+str(seed)+"_"+now.strftime('%Y%m%d%H%M')+".txt", "w",newline='')
    print('samples_per_batch: ',samples_per_batch)

    train_saleerror=np.mean([0])
    train_err = np.mean([0])
    besthr=0
    bestndcg=0
    ifStop = 0
    for i in range(EPOCH_MAX * samples_per_batch):
      model.train()
      users, items, rates, clipList = loader.GetTrainSample(BATCH_SIZE,NEGSAMPLES,K)
      prices=loader.get_salePrice(users,items)
      mask=prices > 0

      cst,slcost = 0, 0 
      infer,inferPrice,CLIPloss = model(torch.tensor(users, dtype=torch.long).to(DEVICE), torch.tensor(items, dtype=torch.long).to(DEVICE),'buy',clipList)
      cst,slcost,l = model.relation_loss(infer, torch.tensor(rates,dtype=torch.float).to(DEVICE), inferPrice, torch.tensor(prices,dtype=torch.float).to(DEVICE), torch.tensor(mask).to(DEVICE),\
                                         torch.tensor(users, dtype=torch.long).to(DEVICE), torch.tensor(items, dtype=torch.long).to(DEVICE))
  
      optimizer.zero_grad()
      # print("L_BUY loss:",l)
      # print("L_I loss:",CLIPloss)
      l = l + gamma*CLIPloss
      l.backward(retain_graph=True) 

      if(UseBidding):
        usersBids, itemsBids, ratesBids = loader.GetBiddingSample(BATCH_SIZE)
        prices=loader.get_bidPrice(usersBids,itemsBids)
        mask=prices > 0

        infer,inferPrice,_ = model(torch.tensor(usersBids, dtype=torch.long).to(DEVICE), torch.tensor(itemsBids, dtype=torch.long).to(DEVICE))
        cst_bid,slcost_bid,l = model.relation_loss_bid(infer, torch.tensor(ratesBids,dtype=torch.float).to(DEVICE), inferPrice, torch.tensor(prices,dtype=torch.float).to(DEVICE), torch.tensor(mask).to(DEVICE),\
                                    torch.tensor(users, dtype=torch.long).to(DEVICE), torch.tensor(items, dtype=torch.long).to(DEVICE))
        l.backward(retain_graph=True) 
        # print("L BID;s :", l)
     
      ### User-CL
      if (UserCL):
        targetI, targetI_buyer, targetI_bidder, U2, U2_bidI = loader._getCLSample(BATCH_SIZE)
        l_graph_cl1 = model._get_cl1_loss(targetI, targetI_buyer, targetI_bidder, U2, U2_bidI)
        l_graph_cl1.backward() 
      optimizer.step()

      errors.append(cst)
      saleerrors.append(slcost)
      if i % samples_per_batch == 0 and (i // samples_per_batch) %2 == 0:
          train_saleerror=np.mean(saleerrors)
          train_err = np.mean(errors)
          model.eval()
          with torch.no_grad():
            testableusers=0
            predictItem = []
            predictUser = []
            for userid in dictUsers: 
              if(len(dictUsers[userid][2])!=0):
                predictItem.append(int(dictUsers[userid][2][0]))
                predictUser.append(userid)
                testableusers+=1

            pred_batch= model._getItemsRating_linear(torch.tensor(predictItem, dtype=torch.long))
            HitRatio,HitRatio20,HitRatio50,NDCG,NDCG20,NDCG50 = metrics.outputPrediction(TOPK,pred_batch,predictUser)
            if HitRatio50 > besthr:
                besthr = HitRatio50
                torch.save(model.state_dict(), PATH)
                ifStop=0
            elif NDCG50 > bestndcg:
                bestndcg = NDCG50
                torch.save(model.state_dict(), PATH)
                ifStop=0
            print("+++++++++++++++++++",model.cl_beta)
            print("{},{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},ifStop:{:d}".format(TIP,i // samples_per_batch, train_err,train_saleerror,HitRatio,HitRatio20,HitRatio50,NDCG,NDCG20,NDCG50, ifStop)) 
            textTrain_file.write("{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},ifStop:{:d}".format(i // samples_per_batch, train_err,train_saleerror,HitRatio,HitRatio20,HitRatio50,NDCG,NDCG20,NDCG50, ifStop) +'\n')            
            textTrain_file.flush()
            ifStop+=1
            if ifStop>200: 
              break
         

def get_UserData():
    df = pd.read_csv('./Data/ebay/DealerFeatures.csv', sep=';', engine='python')
    df=df.sort_values(by=['bidder'])
    print(df.shape)
    del df['bidder']
    values =df.values.astype(np.int32)
    return values
  


print('Device:', torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.set_device(0)

loader = dataloader_ebay.DataLoader(dataset)
USER_NUM, ITEM_NUM = loader.USER_NUM, loader.ITEM_NUM
dictUsers, df_train = loader.dictU, loader.df_train
SEEDs=[1,5,10,20,50]
for i in range(len(SEEDs)):
  seed=SEEDs[i]
  metrics.set_seed(seed)
  PATH = './MRGCN_models/'+FILELABEL+str(seed)
  train(loader, df_train,seed,ItemData=UseItemData,lr=LEARNRATE)