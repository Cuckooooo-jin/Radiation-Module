# -*- coding: UTF-8 -*-
import torch
import numpy as np

class EmbeddingNetwork(torch.nn.Module):
    """The embedding network (encoder) for TimeGAN
    """
    def __init__(self, feature_dim,hidden_dim,num_layers,emb_dim):
        super(EmbeddingNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.emb_dim = emb_dim

        # Embedder Architecture
        self.emb_rnn = torch.nn.GRU(
            input_size=self.feature_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.emb_linear = torch.nn.Linear(self.hidden_dim, self.emb_dim)
        self.emb_sigmoid = torch.nn.Sigmoid()


    def forward(self, X):
        """Forward pass for embedding features from original space into latent space
        Args:
            - X: input time-series features (B x S x F)
            - t: input time-series attributes (B x S x F)
        Returns:
            - H: latent space embeddings (B x S x H)
        """

        # B x 24 x 5
        H_o, H_t = self.emb_rnn(X)
        # B x 24 x 5
        logits = self.emb_linear(H_o)
    
        H = self.emb_sigmoid(logits)
        return H

class RecoveryNetwork(torch.nn.Module):
    """The recovery network (decoder) for TimeGAN
    """
    def __init__(self,emb_dim,feature_dim,hidden_dim,num_layers):
        super(RecoveryNetwork, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        # Recovery Architecture
        self.rec_rnn = torch.nn.GRU(
            input_size=self.emb_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )

        self.outln_continuous = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim,1),
            torch.nn.ReLU()
            )
        self.outln_discrete = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim,10),
            torch.nn.Softmax(dim = 1))
        
    def forward(self, H):
        """Forward pass for the recovering features from latent space to original space
        Args:
            - H: latent representation (B x S x E)
        Returns:
            - X_tilde: recovered data (B x S x F)
        """
        # Dynamic RNN input for ignoring paddings

        H_o, H_t = self.rec_rnn(H)
        
        output_continuous = self.outln_continuous(H_o)
        output_discrete = self.outln_discrete(H_o)
        #X_tilde = self.rec_linear(H_o)
        return torch.cat([output_discrete,output_continuous],dim = 2)

class SupervisorNetwork(torch.nn.Module):
    """The Supervisor network (decoder) for TimeGAN
    """
    def __init__(self, emb_dim,hidden_dim,num_layers):
        super(SupervisorNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.emb_dim = emb_dim

        # Supervisor Architecture
        self.sup_rnn = torch.nn.GRU(
            input_size=self.emb_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers,
            batch_first=True
        )
        self.sup_linear = torch.nn.Linear(self.hidden_dim, self.emb_dim)
        self.sup_relu = torch.nn.LeakyReLU()


    def forward(self, H):
        """Forward pass for the supervisor for predicting next step
        Args:
            - H: latent representation (B x S x E)
        Returns:
            - H_hat: predicted next step data (B x S x E)
        """
        # 365 x 24x 5
        H_o, H_t = self.sup_rnn(H)
        
        # Pad RNN output back to sequence length
        # 365 x 24 x 5
        logits = self.sup_linear(H_o)
        # 365 x 24 x 5
        H_hat = self.sup_relu(logits)
        return H_hat

class GeneratorNetwork(torch.nn.Module):
    """The generator network (encoder) for TimeGAN
    """
    def __init__(self, noise_dim,hidden_dim,num_layers,emb_dim):
        super(GeneratorNetwork, self).__init__()
        self.Z_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.emb_dim = emb_dim

        # Generator Architecture
        self.gen_rnn = torch.nn.GRU(
            input_size=self.Z_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.outln_continuous = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim,1),
            torch.nn.ReLU()
            )
        self.outln_discrete = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim,10),
            torch.nn.Softmax(dim = 1))


    def forward(self, Z):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
        Returns:
            - H: embeddings (B x S x E)
        """
        # b x 24 x 15
        H_o, H_t = self.gen_rnn(Z)
        # b x 24 x 64 --> b x 24 x 5
        output_continuous = self.outln_continuous(H_o)
        output_discrete = self.outln_discrete(H_o)

        return torch.cat([output_discrete,output_continuous],dim = 2)##[b,24,11]

class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN
    """
    def __init__(self,emb_dim, hidden_dim,num_layers):
        super(DiscriminatorNetwork, self).__init__()
        self.input_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Discriminator Architecture
        self.dis_rnn = torch.nn.GRU(
            input_size=self.input_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)

    def forward(self, H):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
        Returns:
            - logits: predicted logits (B x S x 1)
        """
        # 365 x 24 x 5
        H_o, H_t = self.dis_rnn(H)
   
        # 365 x 24
        logits = self.dis_linear(H_o).squeeze(-1)
        return logits
