from torch import nn

class AE(nn.Module):
    def __init__(self, name, encoder, decoder):
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, *args, **kwargs):
        if 'return_z' in kwargs:
            z = self.encoder(x, *args, **kwargs)
            return self.decoder(z, *args, **kwargs), z
        else:
            return self.decoder(self.encoder(x, *args, **kwargs), *args, **kwargs)