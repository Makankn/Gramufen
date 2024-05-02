import torch


def sentence2graph(sent):
  start, end = 0,4 # window length

  occurence = {} # occurence dict: node id: neighbors in window
  feature_matrix =[] #feature matrix
  i2w = {} # word to index dict in window-> word: id
  word_seq=[]
  sent_seq =[]
  i=0
  # constrict w2i and feature matrix
  for word in sent:
    i2w[i] = word
    word_seq.append(word2index[word])
    sent_seq.append(i)
    #feature_matrix.append(embedding_model.wv[word])
    i+=1
  #initializw occurence
  word_seq = torch.tensor(word_seq, dtype=torch.long)
  #feature_matrix = embedding(word_seq)
  for item in sent_seq:
    occurence[item]=set()
  #construct occurence
  while end<=len(sent_seq)+1:
    window = sent_seq[start:end]
    for word in window:
      for innerword in window:
        occurence[word].add(innerword)
    start+=1
    end+=1
  #construct edge_index
  edge_index_1 = []
  edge_index_2 = []
  for source in occurence:
    for dest in occurence[source]:
      edge_index_1.append(source)
      edge_index_2.append(dest)
  edge_index = torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)
  sent_seq = torch.tensor(sent_seq, dtype=torch.long)
  return edge_index, word_seq, sent_seq #torch.tensor(feature_matrix, dtype=torch.float)


