import torch.nn.init as init
import torch
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool
from Sentence2graph import pre_trained_embedding_loader

word_vectors = pre_trained_embedding_loader()

class GCNModel(nn.Module):
  def __init__(self, num_vocabs , gcn_dim, embed_dim, rnn_hidden_dim, rnn_num_layers,
               num_classes, padding_value, bi_directional=True, dropout = 0.5 ):
    super(GCNModel, self).__init__()
    
    #your own emebeding
    # self.embedding = nn.Embedding(num_vocabs, embed_dim)
    
    #pre-Trained embedding
    self.embedding = nn.Embedding.from_pretrained(word_vectors, padding_idx=3000000) 
    self.embedding.weight.requires_grad = False
    self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=rnn_hidden_dim, 
                       bidirectional=bi_directional, batch_first=True, num_layers = rnn_num_layers)
    self.GCN1 = SAGEConv(rnn_hidden_dim*2 if bi_directional else rnn_hidden_dim, gcn_dim, normalize=True)
    self.GCN2 = SAGEConv(gcn_dim, 512, normalize=True)
    self.GCN3 = SAGEConv(512, 512, normalize=True)
    self.linear = nn.Linear(512, num_classes)
    self.dropout = nn.Dropout(dropout)

    self.padding_value=padding_value

    self.relu = nn.ReLU()


    
  def rnn_prepare(self, graph_data, padding_value ):
    batch2nodes ={}
    batch_size = graph_data.y.shape[0]
    for i in range(batch_size):
      batch2nodes[i] = []
    for node, batch in zip(graph_data.word_seq, graph_data.batch):
      batch2nodes[batch.item()].append(node.item())
    seq_lists = [torch.tensor(batch2nodes[item], dtype=torch.long) for item in batch2nodes]
    seq_lens = [len(batch2nodes[item]) for item in batch2nodes]
    seq_lists_padded = pad_sequence(seq_lists, batch_first=True, padding_value=padding_value)
    return seq_lists_padded, seq_lens

  def forward(self, data):
    seq_lists_padded, seq_lens = self.rnn_prepare(graph_data = data, padding_value= self.padding_value)
    seq_lists_padded = seq_lists_padded.to(device)
    seq_features = self.embedding(seq_lists_padded)
    lstm_features, _ = self.rnn(seq_features)
    lstm_features = [lstm_features[i,0:item,:] for i, item in enumerate(seq_lens)]
    x= torch.cat(lstm_features)
    

    edge_index = data.edge_index
    
    x = self.GCN1(x, edge_index)

    x = self.relu(x)

    
    x = self.GCN2(x, edge_index)

    


    x = self.relu(x)

    
    x = self.GCN3(x, edge_index)


    x = self.relu(x)

    x = global_mean_pool(x, data.batch, data.y.shape[0])

    x = self.linear(x)

    return x


