import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """PyTorch port of the published SpecUNet architecture
    (setNetworkMaxpooling.m). With F = num_first_filters, L = num_layers:

      encoder i   Conv3x3(F*2^(i-1)) -> BN -> ReLU -> MaxPool2
                  (skip is taken from the ReLU, before pooling)
      bottleneck  Conv3x3(F*2^L) -> ReLU -> Conv3x3(F*2^L) -> ReLU
                  (bottleneck_bn=True inserts BN after each conv)
      decoder i   ConvT2x2/2(F*2^(L-i)) -> ReLU -> Conv3x3 -> ReLU
                  -> concat[decoder, skip]
                  (no channel reduction: the concatenated tensor feeds the
                  next transposed conv, or the final 1x1 conv, directly)
      output      Conv1x1(1) -> ReLU
    """

    def __init__(self, input_channels, num_layers, num_first_filters, bottleneck_bn=False):
        super().__init__()

        self.encoder = nn.ModuleList()
        f, in_ch, enc_ch = num_first_filters, input_channels, []
        for _ in range(num_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_ch, f, 3, padding=1),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
            ))
            enc_ch.append(f)
            in_ch, f = f, f * 2
        self.pool = nn.MaxPool2d(2, 2)

        # After the loop f = F * 2^L, so the bottleneck is twice as wide as
        # the deepest encoder level, as in the published network.
        def bconv(i, o):
            layers = [nn.Conv2d(i, o, 3, padding=1)]
            if bottleneck_bn:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.ReLU(inplace=True))
            return layers
        self.bottleneck = nn.Sequential(*bconv(in_ch, f), *bconv(f, f))

        self.decoder = nn.ModuleList()
        ch = f
        for i in range(num_layers):
            f //= 2
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(ch, f, 2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, 3, padding=1),
                nn.ReLU(inplace=True),
            ))
            ch = f + enc_ch[-(i + 1)]          # channels after concatenation

        self.final_conv = nn.Sequential(
            nn.Conv2d(ch, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoder, reversed(skips)):
            x = torch.cat([dec(x), skip], dim=1)     # [decoder, skip]
        return self.final_conv(x)