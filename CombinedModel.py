class CombinedModel(nn.Module):
    def __init__(self, gcn_model, image_model):
        super(CombinedModel, self).__init__()
        self.gcn_model = gcn_model
        self.image_model = image_model
        self.projection_head_gcn = ProjectionHead(embedding_dim=512)
        self.projection_head_image = ProjectionHead(embedding_dim=2048)

        self.classifier = Classifier()
        self.classifier_loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
        # self.classifier_loss_function = torch.nn.CrossEntropyLoss()

        self.text_embeddings = None
        self.image_embeddings = None
        self.multimodal_embeddings = None

    def forward(self, gcn_data, image_data, return_embeddings=False):
        gcn_data = gcn_data.to(device)
        image_data = image_data.to(device)
        gcn_output = self.gcn_model(gcn_data)
        image_output = self.image_model(image_data)
        self.text_embeddings = self.projection_head_gcn(gcn_output)
        self.image_embeddings = self.projection_head_image(image_output)
        self.multimodal_embeddings = torch.cat((self.text_embeddings, self.image_embeddings), dim=1)
        
        if return_embeddings:
            return self.multimodal_embeddings  # Return embeddings directly if specified
        
        score = self.classifier(self.multimodal_embeddings)
        probs, output = torch.max(score.data, dim=1)
        similarity = self.text_embeddings @ self.image_embeddings.T

        return output, score



  
