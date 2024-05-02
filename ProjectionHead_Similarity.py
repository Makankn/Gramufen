import torch.nn.functional as F
from torch import nn
import numpy as np


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, 64)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(64)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(2 * 64)
        self.linear_layer = nn.Linear(2 * 64, 128)
        self.gelu = nn.GELU()
        self.drop_out = nn.Dropout(0.3)
        self.layer_norm_2 = nn.LayerNorm(128)
        self.classifier_layer = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = self.linear_layer(x)
        x = self.gelu(x)
        x = self.drop_out(x)
        self.embeddings = x = self.layer_norm_2(x)
        x = self.classifier_layer(x)
        x = self.softmax(x)
        return x



def calculate_loss(model, score, label):
    s_loss = calculate_similarity_loss(model.image_embeddings, model.text_embeddings)
    c_loss = model.classifier_loss_function(score, label)
    loss = 1*c_loss + s_loss
    return loss, c_loss, s_loss


def calculate_similarity_loss(image_embeddings, text_embeddings):
    logits = (text_embeddings @ image_embeddings.T)
    images_similarity = (image_embeddings @ image_embeddings.T)
    texts_similarity = (text_embeddings @ text_embeddings.T)
    targets = F.softmax((images_similarity + texts_similarity) / 2, dim=-1)
    # targets = F.log_softmax((images_similarity + texts_similarity) / 2, dim=-1)

    texts_loss = cross_entropy(logits, targets, reduction='mean')
    images_loss = cross_entropy(logits.T, targets.T, reduction='mean')
    loss = (images_loss + texts_loss) / 2.0
    return loss


def cross_entropy(preds, targets, reduction='none'):
    # entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
    # return entropy(preds, targets)
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
