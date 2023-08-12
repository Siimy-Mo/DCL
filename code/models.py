import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time
import scipy.sparse as sp
from CLIP import *


def to_np(tensor):
  return tensor.cpu().detach().numpy()


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model* 4),
                                          nn.ReLU(True),
                                          nn.Linear(d_model* 4, d_model ))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class DCL(nn.Module):
    def __init__(self, num_users, num_items, adj_mat_buy,adj_mat_bid, w_user, w_item, UseSalePrice,dataset, labels,alpha,beta,embedding_size=128,K=10):
        super(DCL, self).__init__()
        print('=============== Model.py ===============')
        print('self.dev in Model: ', self.dev)
        self.dev = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.num_users = num_users
        self.num_items = num_items
        self.UseSalePrice = UseSalePrice
        self.w_user = w_user.float().to(self.dev)
        self.w_item = w_item.float().to(self.dev)
        self.alpha =alpha
        self.cl_beta = beta
            
        if dataset =='ebay':
            self.item_dense = nn.Linear(15, embedding_size, bias=True) 
        if dataset=='swp':
            self.item_dense = nn.Linear(1049, embedding_size, bias=True) 
        self.user_dense = nn.Linear(num_users, embedding_size, bias=True, device=self.dev) 
        self.price_dense = nn.Linear(embedding_size, 5, bias=True, device=self.dev) 

        torch.nn.init.normal_(self.user_dense.weight, std=0.01) 
        torch.nn.init.normal_(self.item_dense.weight, std=0.01)
        torch.nn.init.normal_(self.price_dense.weight,std=0.01)

        # GCN
        self.BID_cof=nn.Parameter(torch.empty(1,1))
        torch.nn.init.constant_(self.BID_cof,0.5)
        self.BID_mat=adj_mat_bid
        self.BUY_mat=adj_mat_buy
        self.n_layers=1
        self.f = nn.Sigmoid()
        self.initialGraph()

        # CLIP - label encoder
        self.text_token = tokenize(labels).to(self.dev)
        vocab_size = torch.max(self.text_token.cpu()).numpy()+1 
        self.context_length = self.text_token.shape[1]
        transformer_width = embedding_size

        # CLIP - text encoder
        self.text_transformer = Transformer(
            width=transformer_width,
            layers=2,
            heads=8,
            attn_mask=self.build_attention_mask()
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width, device=self.dev) # vocab_size, transformer_width
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.text_ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, num_items))
        if self.text_projection is not None: 
            nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)
        self.item_transformer = Transformer(
            width=embedding_size,
            layers=2,
            heads=8,
            attn_mask=self.build_attention_mask()
        )
        self.item_ln_final = LayerNorm(embedding_size)
        self.item_positional_embedding = nn.Parameter(torch.empty(K,embedding_size))
        nn.init.normal_(self.item_positional_embedding, std=0.01) 
        self.item_projection = nn.Parameter(torch.empty(embedding_size, num_items))
        if self.item_projection is not None: 
            nn.init.normal_(self.item_projection, std=embedding_size ** -0.5)

        self.logit_scale = nn.Parameter(torch.ones([], device=self.dev) * np.log(1 / 0.07))

        self.relation_emb_MLP_BUY = nn.Linear(embedding_size, embedding_size)
        self.relation_emb_MLP_BID = nn.Linear(embedding_size, embedding_size)
        self.PRLoss=torch.nn.BCELoss()


    def initialGraph(self):
        GraphBUY_index,GraphBUY_values,mat_shape = self.generate_adj_mat_single(self.BUY_mat)
        GraphBID_index,GraphBID_values,mat_shape = self.generate_adj_mat_single(self.BID_mat) 
        GraphBID_values = self.alpha * GraphBID_values

        ### 2 - combine BUY and BID graph 
        graph_index = torch.cat((GraphBUY_index,GraphBID_index), 1)
        graph_values = torch.cat((GraphBUY_values,GraphBID_values), 0)
        self.Graph = torch.sparse.FloatTensor(graph_index, graph_values, torch.Size(mat_shape)).coalesce().to(self.dev) 

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return 

    def generate_adj_mat_single(self, mat):
        UI_mat = mat
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = UI_mat.tolil()
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        # _convert_sp_mat_to_sp_tensor
        coo = norm_adj.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return index, data, coo.shape

    def computer(self):
        users_emb = self.user_dense(self.w_user)
        items_emb = self.item_dense(self.w_item)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        g_droped = self.Graph # 4206
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, u_id, i_id, interType='bid',clip_list=[], UReg=0.05,IReg=0.1):
        ############## multi-relation GC
        all_users, all_items = self.computer() 
        U = all_users[u_id] 
        I = all_items[i_id]
        ############## ILCL
        ILCL_loss=[]
        if interType == 'buy':
            text_features = self.textEncoder(i_id)  
            interaction_feature = self.itemEncoder(all_users, clip_list) 

            # normalized features
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            interaction_feature = interaction_feature / interaction_feature.norm(dim=1, keepdim=True)
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits = (logit_scale * interaction_feature.T @ text_features).softmax(dim=-1)

            labels = np.diag(np.ones(logits.shape[1])) 
            labels = torch.tensor(labels,dtype=torch.float).to(self.dev)
            loss_i = F.binary_cross_entropy(logits, labels)
            loss_t = F.binary_cross_entropy(logits.T, labels)
            ILCL_loss = (loss_i + loss_t)/2
        ############## Prediction
        InferInputMF = U * I
        infer = InferInputMF.sum(1)
        plt = self.price_dense(InferInputMF)
        inferPrice = plt.sum(1)
        return infer,inferPrice, ILCL_loss

    def itemEncoder(self, all_users, uid):
        bidders = all_users[uid] 
        x = bidders + self.item_positional_embedding
        x = bidders.permute(1, 0, 2) 
        x = self.item_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.item_ln_final(x)
        x = x[torch.arange(x.shape[0]), bidders.sum(-1).argmax(dim=-1)] @ self.item_projection
        return x
    
    def textEncoder(self, iid):
        text = self.text_token[iid] 
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  
        x = self.text_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.text_ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    def l2_loss(self,t):
        return torch.sum(t ** 2)/2
    
    def relation_loss(self,infer, rate_batch, inferPrice, prices_batch, mask_batch, uid, pids):
        all_users, all_items = self.computer()
        users_emb = all_users[uid]
        items_emb = all_items[pids]

        buy_rel = (users_emb * items_emb).sum(1)
        cost_rel = F.binary_cross_entropy_with_logits(input=buy_rel, target=rate_batch)
        salecost=torch.mean(torch.abs(torch.masked_select(prices_batch,mask_batch)-torch.masked_select(inferPrice,mask_batch)))
        rel_loss=1*cost_rel+self.UseSalePrice*0.00008*salecost
        return to_np(cost_rel), to_np(salecost), rel_loss
    
    def relation_loss_bid(self,infer, rate_batch, inferPrice, prices_batch, mask_batch, uid, pids):
        all_users, all_items = self.computer()
        users_emb = all_users[uid]
        items_emb = all_items[pids]

        bid_rel = (users_emb * items_emb).sum(1)
        cost_rel = F.binary_cross_entropy_with_logits(input=bid_rel, target=rate_batch)
        salecost=torch.mean(torch.abs(torch.masked_select(prices_batch,mask_batch)-torch.masked_select(inferPrice,mask_batch)))
        rel_loss=0.5*(cost_rel+self.UseSalePrice*0.00008*salecost)
        return to_np(cost_rel), to_np(salecost), rel_loss

    def _get_cl1_loss(self, ii, ii_buyer, ii_bidder, ddotU, ij):
        all_users, all_items = self.computer()
        ii = torch.tensor(ii, dtype=torch.long)
        ii_buyer = torch.tensor(ii_buyer, dtype=torch.long)
        ii_bidder = torch.tensor(ii_bidder, dtype=torch.long)
        ddotU = torch.tensor(ddotU, dtype=torch.long)
        ij = torch.tensor(ij, dtype=torch.long)

        pos_item = all_items[ii]
        pos_user = all_users[ii_buyer]
        neg_fine_u = all_users[ii_bidder]
        neg_coarse_u = all_users[ddotU]
        neg_coarse_i = all_items[ij]

        pos_emb = torch.mul(pos_user, pos_item)
        neg_fine = torch.mul(neg_fine_u, pos_item)
        neg_coarse = torch.mul(neg_coarse_u, neg_coarse_i)

        pos_buy_rel = self.relation_emb_MLP_BUY(pos_emb)
        pos_bid_rel = self.relation_emb_MLP_BID(pos_emb)
        neg_fine_rel = self.relation_emb_MLP_BID(neg_fine)
        neg_coarse_rel = self.relation_emb_MLP_BID(neg_coarse)
        #################
        scores_pos = torch.matmul(pos_buy_rel, pos_bid_rel.T).sum(-1)
        scores_neg = torch.matmul(pos_buy_rel, neg_fine_rel.T).sum(-1)
        scores=(scores_pos-scores_neg).sigmoid() # no mask
        loss_CL_fine= self.PRLoss(scores,torch.ones_like(scores,dtype=torch.float32))

        scores_pos = torch.matmul(pos_buy_rel, pos_bid_rel.T).sum(-1)
        scores_neg = torch.matmul(pos_buy_rel, neg_coarse_rel.T).sum(-1)
        scores=(scores_pos-scores_neg).sigmoid() # no mask
        loss_CL_coarse= self.PRLoss(scores,torch.ones_like(scores,dtype=torch.float32))

        # print("L u:", loss_CL_fine + self.cl_beta * loss_CL_coarse )
        return loss_CL_fine + self.cl_beta * loss_CL_coarse
    
    
    def _getItemsRating_linear(self, items):
        all_users, all_items = self.computer() 
        items_emb = all_items[items]
        users_emb = all_users.T
        rating = self.f(torch.matmul(items_emb, users_emb))
        return rating
    