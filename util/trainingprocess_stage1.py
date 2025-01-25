import torch
import torch.optim as optim
from torch.autograd import Variable
from itertools import cycle
from scipy.linalg import norm
from scipy.special import softmax
from sklearn.preprocessing import normalize
from util.dataloader_stage1 import PrepareDataloader
from util.model_regress import Net_encoder, Net_cell
from util.closs_mmd import L1regularization, CellLoss, EncodingLoss
from util.utils import *
import numpy as np
from scipy.spatial import distance_matrix, minkowski_distance, distance

def prepare_input(data_list, config):
    output = []
    for data in data_list:
        output.append(Variable(data.to(config.device)))
    return output


def def_cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class TrainingProcessStage1():
    def __init__(self, config):
        self.config = config
        # load data
        self.train_rna_loaders, self.test_rna_loaders, self.train_atac_loaders, self.test_atac_loaders, self.training_iters = PrepareDataloader(config).getloader()
        self.training_iteration = 0
        for atac_loader in self.train_atac_loaders:
            self.training_iteration += len(atac_loader)
        
        # initialize dataset       
        if self.config.use_cuda:  
            self.model_encoder = torch.nn.DataParallel(Net_encoder(config.input_size).to(self.config.device))
            self.model_cell = torch.nn.DataParallel(Net_cell(config.number_of_class).to(self.config.device))
        else:
            self.model_encoder = Net_encoder(config.input_size).to(self.config.device)
            self.model_cell = Net_cell(config.number_of_class).to(self.config.device)
                
        # initialize criterion (loss)
        self.criterion_cell = CellLoss()
        self.criterion_encoding = EncodingLoss(dim=64, p=config.p, use_gpu = self.config.use_cuda)
        self.l1_regular = L1regularization()
        
        # initialize optimizer (sgd/momemtum/weight decay)
        self.optimizer_encoder = optim.SGD(self.model_encoder.parameters(), lr=self.config.learning_stage1, momentum=self.config.momentum,
                                           weight_decay=0.01)
        self.optimizer_cell = optim.SGD(self.model_cell.parameters(), lr=self.config.learning_stage1, momentum=self.config.momentum,
                                        weight_decay=0.01)


    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.config.learning_stage1 * (0.1 ** ((epoch - 0) // self.config.decay_epoch))
        if (epoch - 0) % self.config.decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def load_checkpoint(self, args):
        if self.config.checkpoint is not None:
            if os.path.isfile(self.config.checkpoint):
                print("=> loading checkpoint '{}'".format(self.config.checkpoint))
                checkpoint = torch.load(self.config.checkpoint)                
                self.model_encoder.load_state_dict(checkpoint['model_encoding_state_dict'])
                self.model_cell.load_state_dict(checkpoint['model_cell_state_dict'])
            else:
                print("=> no resume checkpoint found at '{}'".format(self.config.checkpoint))


    # def calculate_edgelist(self,em1, em2):
    #     #k = 256
    #     #thread = 0.05
    #     #em1 = em1.detach().cpu().numpy()
    #     #em2 = em2.detach().cpu().numpy()
    #     distMat = distance.cdist(em1.view(256,-1).detach().numpy(),em2.view(256,-1).detach().numpy())
    #     #print("distMat_shape",distMat.shape)
    #     normed_disMat = normalize(distMat, axis=1, norm='l2')
    #     print("disMat",distMat)
    #     edgeList=[]
    #     indexArray_count= []
    #     print("disMat_shape",type(normed_disMat))
    #     print("normed_disMat_shape0",normed_disMat)
    #     for i in np.arange(normed_disMat.shape[0]):
    #         indexArray = np.where(normed_disMat[i:] > (-1))
    #         #print("indexArray",len(indexArray[0]))
    #         #print("indexArray_count",indexArray_count)
            
    #         for j in indexArray[0]:
    #             #edgeList.append((i,res[j]))
    #             edgeList.append((i, j))
    #         indexArray_count.append(len(indexArray[0]))
        
    #     #print("indexArray_count",len(indexArray_count))
    #     #print("indexArray_count",indexArray_count)
    #     #count =0
    #     #for k in indexArray_count:
    #     tuple_first = []
    #     tuple_second = []
    #     for k in edgeList:
    #             tuple_first.append(k[0])
    #             tuple_second.append(k[1])
    #     #print("tuple_first",tuple_first)
    #     tuple_first = torch.tensor(tuple_first)
    #     tuple_first = tuple_first.unsqueeze(0)
    #     tuple_second = torch.tensor(tuple_second)
    #     tuple_second = tuple_second.unsqueeze(0)
    #     print("tuple_first",tuple_first)
    #     print("tuple_second",tuple_second.shape)
    #     result = torch.cat((tuple_first,tuple_second),0)
    #     print("result",result)
    #     print("result_size",result.shape)
    #     #torch_edgeList = torch.cat((tuple_first, tuple_second),0)
    #     #print("torch_edgeList", torch_edgeList.shape)       
    #         # count = count+ k
    #     # print("count",count)
    #     #print("edgelist",edgeList)
    #     #print("indexArray",len(indexArray))
    #     return result

    def calculate_edgelist(self, em1, em2):
        # Reshape the tensors
        size = em1.shape[1]
        em1 = em1.view(-1, size)
        em2 = em2.view(-1, size)

        # Convert the tensors to NumPy arrays
        em1 = em1.detach().cpu().numpy()
        em2 = em2.detach().cpu().numpy()

        # Calculate the pairwise distances between the embeddings
        distMat = distance.cdist(em1, em2)
        edgeList = np.zeros(distMat.shape)
        threshold = np.percentile(distMat, 5)
        edgeList[distMat <= threshold] = 1
        edge_index = torch.from_numpy(np.argwhere(edgeList == 1).T).long()
        return edge_index


    def train(self, epoch):
        self.model_encoder.train()
        self.model_cell.train()
        total_encoding_loss, total_cell_loss, total_sample_loss, total_kl_loss = 0., 0., 0., 0.
        total_rna_cell_loss, total_atac_cell_loss= 0., 0.
        self.adjust_learning_rate(self.optimizer_encoder, epoch)
        self.adjust_learning_rate(self.optimizer_cell, epoch)

        # initialize iterator
        iter_rna_loaders = []
        iter_atac_loaders = []
        for rna_loader in self.train_rna_loaders:
            iter_rna_loaders.append(def_cycle(rna_loader))
        for atac_loader in self.train_atac_loaders:
            iter_atac_loaders.append(def_cycle(atac_loader))
                
        for batch_idx in range(self.training_iters):
            # rna forward
            rna_embeddings = []
            rna_cell_predictions = []
            rna_labels = []
            for iter_rna_loader in iter_rna_loaders:
                rna_data, rna_label = next(iter_rna_loader)    
                # prepare data
                rna_data, rna_label = prepare_input([rna_data, rna_label], self.config)
                # model forward
                print(rna_data.shape)
                rna_embedding = self.model_encoder(rna_data)
                edge_index = self.calculate_edgelist(rna_embedding, rna_embedding)
                
                rna_cell_prediction = self.model_cell(rna_embedding, edge_index)
                 
                rna_embeddings.append(rna_embedding)
                rna_cell_predictions.append(rna_cell_prediction)
                rna_labels.append(rna_label)
                
            # atac forward
            atac_embeddings = []
            atac_cell_predictions = []
            atac_labels = []
            for iter_atac_loader in iter_atac_loaders:
                atac_data = next(iter_atac_loader)    
                # prepare data
                atac_data = prepare_input([atac_data], self.config)[0]
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                edgelist = self.calculate_edgelist(atac_embedding, atac_embedding)
                atac_cell_prediction = self.model_cell(atac_embedding,edgelist)

                atac_embeddings.append(atac_embedding)
                atac_cell_predictions.append(atac_cell_prediction)
            
            
       # calculate losses
            rna_cell_loss = sum(self.criterion_cell(pred, label) for pred, label in zip(rna_cell_predictions, rna_labels)) / len(rna_cell_predictions)
            atac_cell_loss = sum(self.criterion_cell(pred, label) for pred, label in zip(atac_cell_predictions, atac_labels)) / len(atac_cell_predictions)

            encoding_loss = self.criterion_encoding(atac_embeddings, rna_embeddings)
            regularization_loss_encoder = self.l1_regular(self.model_encoder)
            regularization_loss_cell = self.l1_regular(self.model_cell)

        # combine losses for multi-task learning
            total_loss = rna_cell_loss + atac_cell_loss + encoding_loss + regularization_loss_encoder + regularization_loss_cell       
            #self.optimizer_encoder.step()
              
            
            
            # update cell weights
            self.optimizer_encoder.zero_grad()
            self.optimizer_cell.zero_grad()
            total_loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_cell.step()
            total_encoding_loss += encoding_loss.data.item()
            total_rna_cell_loss += rna_cell_loss.item()
            total_atac_cell_loss+= torch.tensor(atac_cell_loss).item()
            total_loss_value = total_loss.item()
            progress_bar(batch_idx, self.training_iters,
             'total_loss: %.3f, encoding_loss: %.3f, rna_loss: %.3f, atac_loss: %.3f' % (
                 total_loss_value,
                 total_encoding_loss / (batch_idx + 1),
                 total_rna_cell_loss / (batch_idx + 1),
                 total_atac_cell_loss / (batch_idx + 1),
             ))
        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_cell_state_dict': self.model_cell.state_dict(),
            'model_encoding_state_dict': self.model_encoder.state_dict(),
            'optimizer': self.optimizer_cell.state_dict()            
        })
        
        
    def write_embeddings(self):
        self.model_encoder.eval()
        self.model_cell.eval()
        if not os.path.exists("output/"):
            os.makedirs("output/")
        
        # rna db
        for i, rna_loader in enumerate(self.test_rna_loaders):
            db_name = os.path.basename(self.config.rna_paths[i]).split('.')[0]
            fp_em = open('./output/' + db_name + '_embeddings.txt', 'w')
            fp_pre = open('./output/' + db_name + '_predictions.txt', 'w')
            for batch_idx, (rna_data, rna_label) in enumerate(rna_loader):    
                # prepare data
                rna_data, rna_label = prepare_input([rna_data, rna_label], self.config)
                    
                # model forward
                rna_embedding = self.model_encoder(rna_data)
                edgelist = self.calculate_edgelist(rna_embedding, rna_embedding)
                rna_cell_prediction = self.model_cell(rna_embedding,edgelist)
                           
                rna_embedding = rna_embedding.data.cpu().numpy()
                rna_cell_prediction = rna_cell_prediction.data.cpu().numpy()
                
                # normalization & softmax
                rna_embedding = rna_embedding / norm(rna_embedding, axis=1, keepdims=True)
                rna_cell_prediction = softmax(rna_cell_prediction, axis=1)
                                
                # write embeddings
                test_num, embedding_size = rna_embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(rna_embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(rna_embedding[print_i][print_j]))
                    fp_em.write('\n')
                    
                # write predictions
                test_num, prediction_size = rna_cell_prediction.shape
                for print_i in range(test_num):
                    fp_pre.write(str(rna_cell_prediction[print_i][0]))
                    for print_j in range(1, prediction_size):
                        fp_pre.write(' ' + str(rna_cell_prediction[print_i][print_j]))
                    fp_pre.write('\n')
                
                progress_bar(batch_idx, len(rna_loader),
                         'write embeddings and predictions for db:' + db_name)                    
            fp_em.close()
            fp_pre.close()
        
        
        # atac db
        for i, atac_loader in enumerate(self.test_atac_loaders):
            db_name = os.path.basename(self.config.atac_paths[i]).split('.')[0]
            fp_em = open('./output/' + db_name + '_embeddings.txt', 'w')
            fp_pre = open('./output/' + db_name + '_predictions.txt', 'w')
            for batch_idx, (atac_data) in enumerate(atac_loader):    
                # prepare data
                atac_data = prepare_input([atac_data], self.config)[0]
                
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                edgelist = self.calculate_edgelist(atac_embedding , atac_embedding )
                atac_cell_prediction = self.model_cell(atac_embedding,edgelist)
                                
                                
                atac_embedding = atac_embedding.data.cpu().numpy()
                atac_cell_prediction = atac_cell_prediction.data.cpu().numpy()
                
                # normalization & softmax
                atac_embedding = atac_embedding / norm(atac_embedding, axis=1, keepdims=True)
                atac_cell_prediction = softmax(atac_cell_prediction, axis=1)
                
                # write embeddings
                test_num, embedding_size = atac_embedding.shape
                for print_i in range(test_num):
                    fp_em.write(str(atac_embedding[print_i][0]))
                    for print_j in range(1, embedding_size):
                        fp_em.write(' ' + str(atac_embedding[print_i][print_j]))
                    fp_em.write('\n')
                    
                # write predictions
                test_num, prediction_size = atac_cell_prediction.shape
                for print_i in range(test_num):
                    fp_pre.write(str(atac_cell_prediction[print_i][0]))
                    for print_j in range(1, prediction_size):
                        fp_pre.write(' ' + str(atac_cell_prediction[print_i][print_j]))
                    fp_pre.write('\n')
                
                progress_bar(batch_idx, len(atac_loader),
                         'write embeddings and predictions for db:' + db_name)                    
            fp_em.close()
            fp_pre.close()       
        
