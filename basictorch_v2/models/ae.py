from torch import nn

class AE(nn.Module):
    def __init__(self, name, encoder, decoder):
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, *args, **kwargs):
        return self.decoder(self.encoder(x, *args, **kwargs), *args, **kwargs)