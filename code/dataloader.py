import sys
import pandas as pd
from scipy import sparse
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import random

DATASET=sys.argv[1]

def read_process_ebay(filname, sep="\t"):
    col_names = ["user", "item","price"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    return df

def read_process_swp(filname, sep="\t"):
    df = pd.read_csv(filname, sep=sep, engine='python')
    df = df[["uid", "iid","price"]]
    df.columns = ["user", "item","price"]
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    return df

class DataLoader():
    '''
    Load Movielens-20m dataset
    '''
    def __init__(self, dataset):
        self.datasetType = dataset
        self.testList=[]
        self.short_term_size = 10
        if (dataset == 'ebay'): 
            self.USER_NUM = 3388    
            self.ITEM_NUM = 628
            self._generate_EbayRankData_ebay()
        else:
            self.USER_NUM = 5932
            self.ITEM_NUM = 1171
            self._generate_EbayRankData_swp()
        self.read_SalePrices()
        self.read_BidPrices()
        self.read_Bidding()

        self.dictI = dict()
        self.read_order()
        print('len(self.dictI),len(self.dictU)',len(self.dictI),len(self.dictU))


    def GetTrainSample(self,BatchSize=256,negsamples=1, K=10):
        trainusers=np.asarray([])
        trainitems=np.asarray([])
        traintargets=np.asarray([])
        clipList= []
        numusers=int(BatchSize/(negsamples+1))
        for i in range(numusers):
            fixedBidderLength = [0 for i in range(K)]
            posItemBidders = []

            batchusers=random.choice(list(self.dictU.keys()))
            while len(self.dictU[batchusers][0])==0:
                batchusers=random.choice(list(self.dictU.keys()))

            trainusers=np.append(trainusers,np.repeat(batchusers, negsamples+1))
            ##Pos
            posItem = np.random.choice(self.dictU[batchusers][0], 1) 
            if posItem[0] in self.dictI:
                posItemBidders = self.dictI[posItem[0]][0]
            trainitems=np.append(trainitems,posItem)
            traintargets=np.append(traintargets,[1.0])
            
            fixedBidderLength += posItemBidders
            clipList= clipList + fixedBidderLength[-K:] + fixedBidderLength[-K:]

            ##Neg
            negItem = np.random.choice(self.dictU[batchusers][1], 1)
            while negItem.tolist()[0] not in self.train.keys() or negItem == posItem:
                negItem = np.random.choice(self.dictU[batchusers][1], 1)
            trainitems=np.append(trainitems,negItem)
            traintargets=np.append(traintargets,np.zeros(negsamples))

        clipList = np.reshape(np.array(clipList), (numusers*(1+negsamples),-1))

        return trainusers, trainitems,traintargets, clipList

    def GetTrainBUYSample(self,BatchSize=256,negsamples=1):
        trainusers=np.asarray([])
        trainitems=np.asarray([])
        traintargets=np.asarray([])
        trainbidderorder=[]
        trainpriceorder=[]
        numusers=int(BatchSize/(negsamples+1))
        for i in range(numusers):
            batchusers=random.choice(list(self.dictU.keys()))
            while len(self.dictU[batchusers][0])==0:
                batchusers=random.choice(list(self.dictU.keys()))

            trainusers=np.append(trainusers,np.repeat(batchusers, negsamples+1))
            ##Pos
            posItem = np.random.choice(self.dictU[batchusers][0], 1)
            trainitems=np.append(trainitems,posItem)
            traintargets=np.append(traintargets,[1.0])
            
            trainbidderorder= trainbidderorder + self.train[posItem.tolist()[0]][2]
            trainpriceorder= trainpriceorder + self.train[posItem.tolist()[0]][3]

            ##Neg
            negItem = np.random.choice(self.dictU[batchusers][1], 1)
            while negItem.tolist()[0] not in self.train.keys() or negItem == posItem:
                negItem = np.random.choice(self.dictU[batchusers][1], 1)
            trainitems=np.append(trainitems,negItem)
            traintargets=np.append(traintargets,np.zeros(negsamples))

            trainbidderorder= trainbidderorder + self.train[negItem.tolist()[0]][2]
            trainpriceorder= trainpriceorder + self.train[negItem.tolist()[0]][3]

        trainbidderorder = np.reshape(np.array(trainbidderorder), (numusers*(1+negsamples),-1))
        trainpriceorder = np.reshape(np.array(trainpriceorder), (numusers*(1+negsamples),-1))
        return trainusers, trainitems,traintargets, trainbidderorder, trainpriceorder

    def GetBiddingSample_npU(self, BatchSize=256,negsamples=1):
        trainusers=np.asarray([])
        trainitems=np.asarray([])
        traintargets=np.asarray([])
        numusers=int(BatchSize/(negsamples+1))
        for i in range(numusers):
            batchitems=random.choice(list(self.dictI.keys()))
            while len(self.dictI[batchitems][0])==0:
                batchitems=random.choice(list(self.dictI.keys()))

            trainitems=np.append(trainitems,np.repeat(batchitems, negsamples+1))
            ##Pos
            posUser = np.random.choice(self.dictI[batchitems][0], 1)
            trainusers=np.append(trainusers,posUser)
            traintargets=np.append(traintargets,[1.0])
        
            ##Neg
            negUser = np.random.choice(self.dictI[batchitems][1], 1)
            while negUser.tolist()[0] not in self.train.keys() or negUser == posUser:
                negUser = np.random.choice(self.dictI[batchitems][1], 1)
            trainusers=np.append(trainusers,negUser)
            traintargets=np.append(traintargets,np.zeros(negsamples))
            
        return trainusers,trainitems,traintargets


    def _get_order(self,iid):
        return self.test[iid][2]

    def GetBiddingSample(self, BatchSize=256,negsamples=1):
        trainusers=np.asarray([])
        trainitems=np.asarray([])
        traintargets=np.asarray([])
        numusers=int(BatchSize/(negsamples+1))
        for i in range(numusers):
            batchusers=random.choice(list(self.BIDDINGDATA.keys())) 
            while len(self.BIDDINGDATA[batchusers])==0:
                batchusers=random.choice(list(self.BIDDINGDATA.keys()))

            trainusers=np.append(trainusers,np.repeat(batchusers, negsamples+1))
            ##Pos
            trainitems=np.append(trainitems,np.random.choice(self.BIDDINGDATA[batchusers], 1))
            traintargets=np.append(traintargets,[1.0])
            ##Neg
            negitem=random.randint(0,self.ITEM_NUM-1)
            while negitem in self.BIDDINGDATA[batchusers]:
                negitem=random.randint(0,self.ITEM_NUM-1)
            trainitems=np.append(trainitems,negitem)
            traintargets=np.append(traintargets,np.zeros(negsamples))
            
        return trainusers,trainitems,traintargets

    def get_salePrice(self, userarr,itemarr):
        prices=self.SALEDATA[userarr[:], itemarr[:]]
        return np.asarray(prices).reshape(-1)
    
    def get_bidPrice(self, userarr,itemarr):
        prices=self.SALEDATA_bid[userarr[:], itemarr[:]]
        return np.asarray(prices).reshape(-1)

    def _getCLSample(self,BatchSize=256):
        targetI=np.asarray([])
        targetI_buyer=np.asarray([])
        targetI_bidder = np.asarray([])
        U2=np.asarray([])
        U2_bidI=np.asarray([])

        numusers=int(BatchSize)
        for i in range(numusers):
            batchitems=random.choice(list(self.dictI.keys()))
            buyer = self.dictI[batchitems][2]
            while len(self.dictI[batchitems][0])==0 or len(self.dictU[buyer][2])==0:
                batchitems=random.choice(list(self.dictI.keys()))
                buyer = self.dictI[batchitems][2]
            
            targetI=np.append(targetI,np.repeat(batchitems, 1))
            # Neg
            bidder = np.random.choice(self.dictI[batchitems][0], 1) 

            targetI_buyer=np.append(targetI_buyer,buyer)
            targetI_bidder=np.append(targetI_bidder,bidder)

            bidder_2=random.choice(list(self.dictU.keys()))
            while bidder_2 == buyer or len(self.dictU[bidder_2][0])==0:
                bidder_2=random.choice(list(self.dictU.keys()))
            
            bidder_2_I = random.choice(list(self.dictU[bidder_2][0])) 
            U2=np.append(U2,bidder_2)
            U2_bidI=np.append(U2_bidI,bidder_2_I)
        return targetI, targetI_buyer, targetI_bidder, U2, U2_bidI


########## GGGGGGGGGGraph
    def _generate_Graph(self, dictUsers):
        data_user = dict()
        for uid in self.dictU:
            users = uid         # [uid]
            if len(self.dictU[uid][0])>0: 
                item_id = self.dictU[uid][0]# iid
            data_user[users] = item_id
        # BID
        col_names = ["uid", "iid","bid"]
        data = pd.read_csv("./Data/"+self.datasetType+"/Biddings.csv", sep=';', header=None, names=col_names, engine='python') 
        data_grouped_by_item = data.groupby('iid')
        data_item = dict()
        for _, group in data_grouped_by_item:
            item_id = int(group['iid'].tolist()[0]) 
            users = group['uid'].tolist()   
            data_item[item_id] = users

        adj_mat_buy,adj_mat_bid = self.Data_2_Graph_2(data_user, data_item)
        print('Interact matrix:', adj_mat_buy.shape,adj_mat_bid.shape)

        return adj_mat_buy,adj_mat_bid

    def Data_2_Graph_2B(self, data_user, data_item):
        adj_mat = sp.dok_matrix((self.USER_NUM, self.ITEM_NUM), dtype=np.float32)
        ### BUY
        for uid in data_user:
            uids_idx = uid
            item_idx = [int(uid) for uid in data_user[uid]]

            for j in range(len(item_idx)):
                if item_idx[j] == -1:
                    continue
                else:
                    adj_mat[uids_idx, item_idx[j]] = 1.

        ### BID
        for pid in data_item:
            item_idx = pid
            uids_idx = [int(uid) for uid in data_item[pid]]

            for j in range(len(uids_idx)):
                if uids_idx[j] == -1:
                    continue
                else:
                    adj_mat[uids_idx[j], item_idx] = 1.

        print('edges number in train/test bid node:', adj_mat)
        return adj_mat

    def Data_2_Graph_2(self, data_user, data_item):
        adj_mat1 = sp.dok_matrix((self.USER_NUM, self.ITEM_NUM), dtype=np.float32)
        adj_mat2 = sp.dok_matrix((self.USER_NUM, self.ITEM_NUM), dtype=np.float32)
        for uid in data_user:
            uids_idx = uid
            item_idx = [int(uid) for uid in data_user[uid]]
            for j in range(len(item_idx)):
                if item_idx[j] == -1:
                    continue
                else:
                    adj_mat1[uids_idx, item_idx[j]] = 1.
        ### BID
        for pid in data_item:
            item_idx = pid
            uids_idx = [int(uid) for uid in data_item[pid]]

            for j in range(len(uids_idx)):
                if uids_idx[j] == -1:
                    continue
                else:
                    adj_mat2[uids_idx[j], item_idx] = 1.

        print('count_nonzero:', adj_mat1.count_nonzero(),adj_mat2.count_nonzero())
        return adj_mat1,adj_mat2

    def read_SalePrices(self):    
        col_names = ["user", "item","price"]
        df = pd.read_csv("./Data/"+self.datasetType+"/Ratings.csv", sep=';', header=None, names=col_names, engine='python')
        for col in ("user", "item"):
            df[col] = df[col].astype(np.int32)
        SALEDATA=csr_matrix( (self.USER_NUM,self.ITEM_NUM) )  
        for index, row in df.iterrows():
            userid=int(row['user'])
            itemid=int(row['item'])
            saleprice=row['price']
            SALEDATA[userid,itemid]=saleprice
        self.SALEDATA = SALEDATA

    def read_BidPrices(self):    
        col_names = ["user", "item","price"]
        df = pd.read_csv("./Data/"+self.datasetType+"/Biddings.csv", sep=';', header=None, names=col_names, engine='python')
        for col in ("user", "item"):
            df[col] = df[col].astype(np.int32)
        SALEDATA=csr_matrix( (self.USER_NUM,self.ITEM_NUM) )  
        for index, row in df.iterrows():
            userid=int(row['user'])
            itemid=int(row['item'])
            saleprice=row['price']
            SALEDATA[userid,itemid]=saleprice
        self.SALEDATA_bid = SALEDATA

    def read_Bidding(self):
        col_names = ["user", "item","bid"]
        dfBidding = pd.read_csv("./Data/"+self.datasetType+"/Biddings.csv", sep=';', header=None, names=col_names, engine='python') 
        BiddingDict={}
        BiddingMatrixList=[]
        for index, row in dfBidding.iterrows():
            userid=int(row['user'])
            itemid=int(row['item'])
            if userid in self.dictU :
                BiddingMatrixList.append([userid,itemid,0.5])
                if len(self.dictU[userid][2])==1:
                    if userid in BiddingDict:
                        if itemid != self.dictU[userid][2][0]:
                            BiddingDict[userid].append(itemid)
                    else:
                        BiddingDict[userid]=list()
                        if itemid != self.dictU[userid][2][0]:
                            BiddingDict[userid].append(itemid)
                else:
                    if userid in BiddingDict:
                        BiddingDict[userid].append(itemid)
                    else:
                        BiddingDict[userid]=list()
                        BiddingDict[userid].append(itemid)
        print(len(BiddingDict))
        BiddingMatrix=np.asarray(BiddingMatrixList)
        np.savetxt("./Data/"+self.datasetType+"/Biddings.txt", BiddingMatrix, delimiter=";")
        self.BIDDINGDATA = BiddingDict

    def read_order(self):
        col_names = ["uid", "iid","bid"]
        data = pd.read_csv("./Data/"+self.datasetType+"/Biddings.csv", sep=';', header=None, names=col_names, engine='python')

        data_item = []
        data_grouped_by_item = data.groupby('iid')

        short_term_size = self.short_term_size
        added =[]
        for _, group in data_grouped_by_item:
            iid = int(group['iid'].tolist()[0])

            if iid in self.item_users:
                # 0->TrainPos,2->Neg users, 2->TestPos, 3->attr
                self.dictI[iid]={0:list(),1:list(),2:list()}
                self.dictI[iid][0]=group['uid'].tolist()
                self.dictI[iid][2]=self.item_users[iid][0]

                for i in range(self.USER_NUM):
                    if i not in self.dictI[iid][0] and i !=self.dictI[iid][2]:
                        self.dictI[iid][1].append(i)####111

                maxPrice = self.item_users[iid][1]
                group['score'] = group.apply(lambda x: 1+(x.bid - maxPrice)/maxPrice,axis=1)
                
                current_pid = iid
                current_uid = self.item_users[iid][0]
                bidder_list = []
                bid_list = []
                for k in range(0, short_term_size):
                    bidder_list.append(-1)
                    bid_list.append(-1)
                for k in range(1,len(group['uid'].tolist())+1):
                    bidder_list.append(int(group['uid'].tolist()[-k]))
                    bid_list.append(group['score'].tolist()[-k])
                bidder_list = bidder_list[-short_term_size:] 
                bid_list = bid_list[-short_term_size:]
                data_item.append((current_pid, current_uid, bidder_list, bid_list))
                added.append(current_pid)


        for iid in self.item_users:
            bidder_list = []
            bid_list = []
            if iid not in added:
                current_pid = iid
                current_uid = self.item_users[iid][0]
                for k in range(0, short_term_size):
                    bidder_list.append(-1)
                    bid_list.append(-1)
                data_item.append((current_pid, current_uid, bidder_list, bid_list))

        print('total order Num, added orders:', len(data_item), len(added))
        te_items = self.testList
        train, test = dict(), dict()
        for i in range(len(data_item)):
            if data_item[i][0] in te_items:
                test[data_item[i][0] ]=data_item[i]
            else:
                train[data_item[i][0] ]=data_item[i]

        print('Data length of train/test dataset:', len(train), len(test)) # 191 20
        self.train, self.test = train, test
        itemRange = self.df_train['item'].tolist()
        self.diff = []
        for i in range(len(itemRange)):
            if itemRange[i] not in self.dictI.keys():
                self.diff.append(itemRange[i])


    def _get_ItemData(self):
        df = pd.read_csv('./Data/'+self.datasetType+'/ItemFeatures.csv', sep=';', engine='python')
        df=df.sort_values(by=['auctionid'])
        del df['auctionid']
        df=pd.concat([df,df['item'].str.get_dummies(sep=' ').add_prefix('Name_').astype('int8')],axis=1)    
        df=pd.concat([df,df['auction_type'].str.get_dummies(sep=' ').add_prefix('Auction_').astype('int8')],axis=1) 
        labels = df['item'].tolist()
        del df['item']
        del df['auction_type']
        df=pd.get_dummies(df,dummy_na=True)
        df=df.fillna(df.mean())
        df=df.dropna(axis=1, how='all')
        values=df.values
        return values,labels

    def _generate_EbayRankData_ebay(self):
        print('---- read eBay dataset')
        df = read_process_ebay("./Data/"+self.datasetType+"/Ratings.csv", sep=";")
        dictUsers={}
        self.item_users = dict()
        BuyMatrixList=[]
        testMatrixList = []
        
        ##Filling all PosTrain
        print('Filling all PosTrain')
        for index, row in df.iterrows():
            userid=int(row['user'])
            if userid not in dictUsers:
                # 0->PosTrain,1->NegTrain,2->TestPos 1,3->Test 100,4->ValidationPos,5->Validation 100
                dictUsers[userid]={0:list(),1:list(),2:list()}
            dictUsers[userid][0].append(row['item'])    ####000
            self.item_users[row['item']] = [userid, row['price']]
            BuyMatrixList.append([userid,row['item'],1.])

        print('\nFilling Neg Instance ')    
        for userid in dictUsers: 
            for i in range(self.ITEM_NUM):
                if i not in dictUsers[userid][0] :
                    dictUsers[userid][1].append(i)####111

        print('\nSelecting Test Instance ')   
        uidCanTest = []
        for userid in dictUsers: 
            if(len(dictUsers[userid][0])>1):
                lastitem = dictUsers[userid][0].pop(len(dictUsers[userid][0])-1)
                dictUsers[userid][2].append(lastitem)####222
                self.testList.append(lastitem)
                uidCanTest.append(userid)
                testMatrixList.append([userid,lastitem])
        print('\nThe length of users who have more than 2 pos sample(Test): ', len(uidCanTest))

        ##All Training Positive Data
        print('All Training Positive Data')
        TrainItems=list()
        TrainUsers=list()
        TrainTargets=list()
        for userid in dictUsers:
            ItemsLength=len(dictUsers[userid][0]) 
            TrainUsers.extend(np.repeat(userid, ItemsLength))
            TrainItems.extend(dictUsers[userid][0])
            TrainTargets.extend(np.repeat(1.0, ItemsLength))
        
        df_train= pd.DataFrame(columns=['user', 'item', 'rate'])
        df_train['user']=TrainUsers
        df_train['item']=TrainItems
        df_train['rate']=TrainTargets

        self.dictU = dictUsers
        self.df_train = df_train
        BuyMatrix=np.asarray(BuyMatrixList)
        testMatrix=np.asarray(testMatrixList)
        np.savetxt("./Data/"+self.datasetType+"/allBuy.txt", BuyMatrix, delimiter=";")
        np.savetxt("./Data/"+self.datasetType+"/test.txt", testMatrix, delimiter=";")

    def _generate_EbayRankData_swp(self):
        print('---- read swp dataset')
        df3 = read_process_swp("./Data/MFtrainBUY3.csv", sep=";")
        df4 = read_process_swp("./Data/MFtestBUY4.csv", sep=";")
        dictUsers={}
        self.item_users = dict()
        BuyMatrixList=[]
        testMatrixList = []

        ##Filling all PosTrain
        print('Filling all PosTrain (Train BUY)')
        for index, row in df3.iterrows():
            userid=int(row['user'])
            itemid=int(row['item'])
            if userid not in dictUsers:
                # 0->TrainPos,2->Neg Items, 2->TestPos
                dictUsers[userid]={0:list(),1:list(),2:list()}
            dictUsers[userid][0].append(itemid)####000
            self.item_users[row['item']] = [userid, row['price']]
            BuyMatrixList.append([userid,row['item'],1.])
        ##Filling all PosTrain -end-

        print('\nSelecting Test Instance (TestBUY)') 
        ## Selecting Test Instance    
        for index, row in df4.iterrows():
            userid=int(row['user'])
            itemid=int(row['item'])
            if userid not in dictUsers:
                # 0->TrainPos,2->Neg Items, 2->TestPos
                dictUsers[userid]={0:list(),1:list(),2:list()}
            dictUsers[userid][2].append(itemid)####000
            self.item_users[row['item']] = [userid, row['price']]
            testMatrixList.append([userid,itemid])
        print('具有训练/预测集购买交互的用户数量：', len(dictUsers))

        print('\nFilling Neg Instance ')
        ## Filling Neg Instance          
        for userid in dictUsers: 
            for i in range(self.ITEM_NUM):
                if i not in dictUsers[userid][0] and  i not in dictUsers[userid][2]:
                    dictUsers[userid][1].append(i)####111

        ##All Training Positive Data
        print('All Training Positive Data')
        TrainItems=list()
        TrainUsers=list()
        TrainTargets=list()
        for userid in dictUsers:
            ItemsLength=len(dictUsers[userid][0])
            TrainUsers.extend(np.repeat(userid, ItemsLength))
            TrainItems.extend(dictUsers[userid][0])
            TrainTargets.extend(np.repeat(1.0, ItemsLength)) 

        df_train= pd.DataFrame(columns=['user', 'item', 'rate'])
        df_train['user']=TrainUsers
        df_train['item']=TrainItems
        df_train['rate']=TrainTargets

        self.dictU = dictUsers
        self.df_train = df_train
        BuyMatrix=np.asarray(BuyMatrixList)
        testMatrix=np.asarray(testMatrixList)
        np.savetxt("./Data/"+self.datasetType+"/allBuy.txt", BuyMatrix, delimiter=";")
        np.savetxt("./Data/"+self.datasetType+"/test.txt", testMatrix, delimiter=";")

    

def dataPrep():
    if (DATASET == 'ebay'):
        df=pd.read_csv("./Data/"+DATASET+"/auction.csv", sep=',', engine='python')
        df.bidder=df.bidder.astype(str)
        auctionidEncoder= preprocessing.LabelEncoder()
        BidderEncoder= preprocessing.LabelEncoder()
        df.auctionid= auctionidEncoder.fit_transform(df.auctionid)
        df.bidder= BidderEncoder.fit_transform(df.bidder)
        df.bidder=df.bidder.astype(int)
        df.auctionid=df.auctionid.astype(int)
        dfItemFeatures=df[['auctionid','openbid','item','auction_type']].drop_duplicates(subset='auctionid')
    else:
        df_=pd.read_csv("./Data/MFdata.csv", header=0, sep=';', engine='python').dropna(how='any') #287050
        df = df_[['iid','uid','score','winner']]
        df_meta = df_[['iid','item','auction_type','openbid','winner','price']]

        df.columns = ['auctionid','bidder','bid','winner']
        df_meta.columns = ['auctionid','item','auction_type','openbid','bidder','price']
        dfItemFeatures=df_meta[['auctionid','openbid','item','auction_type']].drop_duplicates(subset='auctionid')

    dfItemFeatures.auctionid=dfItemFeatures.auctionid.astype(int)
    dfDealerFeatures= df[['bidder']].drop_duplicates(subset='bidder')
    dfDealerFeatures.bidder=dfDealerFeatures.bidder.astype(int)

    df=df.sort_values(by=['auctionid','bid'],ascending=False)

    PurchasedItems=[]
    PurchaseMatrixList=[]
    BiddingMatrixList=[]

    if (DATASET == 'ebay'):
        for index, row in df.iterrows():
            auctionid=int(row['auctionid'])
            bid=float(row['bid'])
            price=float(row['price'])
            bidder=int(row['bidder'])
            if(bid==price) and (auctionid not in PurchasedItems):
                PurchasedItems.append(auctionid)
                PurchaseMatrixList.append([bidder,auctionid,price])
            else:
                BiddingMatrixList.append([bidder,auctionid,bid]) 

    else:
        for index, row in df.iterrows():
            auctionid=int(row['auctionid'])
            bid=float(row['bid'])
            winnerflg=int(row['winner'])
            bidder=int(row['bidder'])
            if(winnerflg == 1) and (auctionid not in PurchasedItems):
                PurchasedItems.append(auctionid)
                PurchaseMatrixList.append([bidder,auctionid,bid])
            else:
                BiddingMatrixList.append([bidder,auctionid,bid])

    PurchaseMatrix=np.asarray(PurchaseMatrixList)
    BiddingMatrix=np.asarray(BiddingMatrixList)
    print(PurchaseMatrix.shape)
    print(BiddingMatrix.shape)
    print(dfDealerFeatures.shape)
    print(dfItemFeatures.shape)


    np.savetxt("./Data/"+DATASET+"/Ratings.csv", PurchaseMatrix, delimiter=";")
    np.savetxt("./Data/"+DATASET+"/Biddings.csv", BiddingMatrix, delimiter=";")
    dfItemFeatures.to_csv("./Data/"+DATASET+"/ItemFeatures.csv", sep=';', encoding='utf-8', index=False)
    dfDealerFeatures.to_csv("./Data/"+DATASET+"/DealerFeatures.csv", sep=';', encoding='utf-8', index=False)

if __name__ == '__main__':
    dataPrep()
    print("Done!")