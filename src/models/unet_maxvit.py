import torch
import torch.nn as nn
import timm

class UNet(nn.Module):
    def __init__(self, backbone_name='maxvit_small_tf_512.in1k', pretrained=True, num_classes=3):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)

        # encoder outputs
        channels = [f['num_chs'] for f in self.encoder.feature_info]
        self.center = nn.Conv2d(channels[-1], 512, 3, padding=1)

        self.decoder4 = self._decoder_block(512 + channels[-2], 256)
        self.decoder3 = self._decoder_block(256 + channels[-3], 128)
        self.decoder2 = self._decoder_block(128 + channels[-4], 64)
        self.decoder1 = self._decoder_block(64 + channels[-5], 32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        enc_feats = self.encoder(x)
        x = self.center(enc_feats[-1])

        x = self.decoder4(torch.cat([x, enc_feats[-2]], dim=1))
        x = self.decoder3(torch.cat([x, enc_feats[-3]], dim=1))
        x = self.decoder2(torch.cat([x, enc_feats[-4]], dim=1))
        x = self.decoder1(torch.cat([x, enc_feats[-5]], dim=1))

        return self.final(x)