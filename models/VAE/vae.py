import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.mean = nn.Linear(latent_dim, 2)
        self.var = nn.Linear(latent_dim, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self._init_weights()

    def encode(self, x):
        out = self.encoder(x)
        mean, var = self.mean(out), self.var(out)
        return mean, var

    def reparameterize(self, mean, var):
        eps = torch.randn_like(var)
        return mean + eps * var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, var = self.encode(x)
        z = self.reparameterize(mean, var)
        recon_x = self.decode(z)
        return recon_x, mean, var

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
