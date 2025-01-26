import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from resnet1 import resnet18  

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def pos_embed(self, src, pos):
        assert src.shape == pos.shape, f"Shape mismatch: src shape {src.shape}, pos shape {pos.shape}"
        return src + pos

    def forward(self, src, pos):
        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + (self.dropout(src2) if self.training else src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))) if self.training else F.relu(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
        return src

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        if self.norm is not None:
            output = self.norm(output)
        return output

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFusion, self).__init__()

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14x14 -> 28x28
        self.upsample_2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 7x7 -> 28x28
        self.conv_fuse = DepthwiseSeparableConv(896, in_channels)  # 修改输入通道数为 896
        self.downsample = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)  # 28x28 -> 7x7

    def forward(self, x2, x3, x4):
        x3_up = self.upsample_1(x3)
        x4_up = self.upsample_2(x4)
        fused_features = torch.cat([x2, x3_up, x4_up], dim=1)  
        fused_features = self.conv_fuse(fused_features)
        fused_features = self.downsample(fused_features)

        return fused_features
   
class SharedAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SharedAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        self.channel_conv = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.channel_norm = nn.LayerNorm(in_channels // reduction)
        self.spatial_conv = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.spatial_norm = nn.LayerNorm(in_channels // reduction)
        self.eca = nn.Conv1d(in_channels // reduction, 1, kernel_size=3, padding=1, bias=False)
        self.conv_fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, is_last_layer=False):
        if x.dim() == 2:
            return x

        batch_size, channels, height, width = x.size()
        print(f"SharedAttention input shape: {x.shape}")
        x_channel = x.view(batch_size, channels, -1)
        x_channel = x_channel.permute(0, 2, 1)
        print(f"x_channel shape: {x_channel.shape}")
        Qc = self.channel_conv(x_channel)
        Ac = F.softmax(Qc, dim=1)
        Ac = self.channel_norm(Ac)
        Fout1 = torch.bmm(x_channel, Ac).view(batch_size, channels, height, width)
        Fout1 += x  
        if is_last_layer:
            x_spatial = x.view(batch_size, channels, -1)
            Qs = self.spatial_conv(x_spatial)
            As = F.softmax(Qs, dim=2)
            As = self.spatial_norm(As)
            Fout2 = torch.bmm(As, x_spatial).view(batch_size, channels, height, width)
            Fout2 += x  
            Fout = torch.cat([Fout1, Fout2], dim=1)  # [batch, 2*C, H, W]
        else:
            Fout = Fout1

        Fout = self.eca(Fout.view(batch_size, Fout.size(1), -1)).view(batch_size, Fout.size(1), height, width)
        Fout = self.conv_fuse(Fout)
        Fout = self.dropout(Fout) if self.training else Fout

        print(f"SharedAttention output shape: {Fout.shape}")

        return Fout


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, quantile=0.5):
        super(TransformerModel, self).__init__()

        maps = 128  
        nhead = 8
        dim_feedforward = 512
        dropout = 0.1
        num_layers = 6  

        self.base_model = resnet18(pretrained=False, maps=maps)

        self.multi_scale_fusion = MultiScaleFusion(maps)

        self.depthwise_separable_conv = DepthwiseSeparableConv(maps * 2, maps)  
        encoder_layer = TransformerEncoderLayer(maps, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(maps)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))  

        self.pos_embedding = FixedPositionalEncoding(maps)

        self.shared_attention = SharedAttention(maps)  

        self.feed = nn.Linear(maps, 2)  

        self.loss_op = nn.L1Loss()

    def forward(self, x_in):
        device = x_in["face"].device  

        original_features, x2, x3, x4 = self.base_model(x_in["face"])

        fused_features = self.multi_scale_fusion(x2, x3, x4)

        combined_features = torch.cat((fused_features, original_features), dim=1)
        combined_features = self.depthwise_separable_conv(combined_features)
        batch_size = combined_features.size(0)
        combined_features = combined_features.flatten(2)  # [batch, channel, height, width] -> [batch, channel, H*W]
        combined_features = combined_features.permute(2, 0, 1)  # [batch, channel, H*W] -> [H*W, batch, channel]

        cls_token = self.cls_token.expand(1, batch_size, -1).to(device)  
        combined_features = torch.cat((cls_token, combined_features), dim=0)  


        pos_encoded = self.pos_embedding(combined_features)  


        transformer_out = self.encoder(pos_encoded, pos_encoded)


        cls_output = transformer_out[0]  


        attention_output = self.shared_attention(cls_output, is_last_layer=True)


        output = self.feed(attention_output)
        return output 

    def loss(self, x_in, label):
        prediction = self.forward(x_in)

        loss = self.loss_op(prediction, label)
        return loss