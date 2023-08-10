import timm
import torch
from torch import nn
from torch.nn import functional as F

from src.layer.arcface_adaptive_margin import (
    ArcMarginProduct,
    ArcMarginProduct_subcenter,
)
from src.layer.gem import GeM


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations=[6, 12, 18]):
        super(MultiAtrousModule, self).__init__()

        self.d0 = nn.Conv2d(
            in_chans, 512, kernel_size=3, dilation=dilations[0], padding="same"
        )
        self.d1 = nn.Conv2d(
            in_chans, 512, kernel_size=3, dilation=dilations[1], padding="same"
        )
        self.d2 = nn.Conv2d(
            in_chans, 512, kernel_size=3, dilation=dilations[2], padding="same"
        )
        self.conv1 = nn.Conv2d(512 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return x


class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20)  # use default setting.

    def forward(self, x):
        """
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score
        """
        x = self.conv1(x)
        x = self.bn(x)

        feature_map_norm = F.normalize(x, p=2, dim=1)

        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)

        x = att * feature_map_norm
        return x, att_score


class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):
        bs, c, w, h = fl.shape

        fl_dot_fg = torch.bmm(fg[:, None, :], fl.reshape(bs, c, -1))
        fl_dot_fg = fl_dot_fg.reshape(bs, 1, w, h)
        fg_norm = torch.norm(fg, dim=1)

        fl_proj = (fl_dot_fg / fg_norm[:, None, None, None]) * fg[:, :, None, None]
        fl_orth = fl - fl_proj

        f_fused = torch.cat([fl_orth, fg[:, :, None, None].repeat(1, 1, w, h)], dim=1)
        return f_fused


class Dolg(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg
            - n_classes => クラス数
            - model_name => モデル名
            - pretrained => 学習済みモデル使用有無
            - embedding_size => 埋め込み次元サイズ
            - use_product => モデル内部でArcface用のコサイン類似度計算を実施するか
            - dilations => 3つのdilationsの設定. ex: [6, 12, 18]
        """
        super(Dolg, self).__init__()

        self.cfg = cfg
        self.n_classes = self.cfg.n_classes
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            features_only=True,
        )

        backbone_out = self.backbone.feature_info[-1]["num_chs"]
        backbone_out_1 = self.backbone.feature_info[-2]["num_chs"]

        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g

        self.global_pool = GeM()

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = cfg.embedding_size

        self.neck = nn.Sequential(
            nn.Linear(fusion_out, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU(),
        )

        self.head_in_units = self.embedding_size
        if self.cfg.use_product == "normal":
            self.head = ArcMarginProduct(self.embedding_size, self.n_classes)
        elif self.cfg.use_product == "sub-center":
            self.head = ArcMarginProduct_subcenter(self.embedding_size, self.n_classes)
        else:
            self.head = nn.Identity()

        self.mam = MultiAtrousModule(
            backbone_out_1, feature_dim_l_g, self.cfg.dilations
        )
        self.conv_g = nn.Conv2d(backbone_out, feature_dim_l_g, kernel_size=1)
        self.bn_g = nn.BatchNorm2d(
            feature_dim_l_g,
            eps=0.001,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.act_g = nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()

    def forward(self, x):
        x_emb = self.extract_features(x)

        logits = self.head(x_emb)
        return logits

    def extract_features(self, x):
        x = self.backbone(x)

        x_l = x[-2]
        x_g = x[-1]

        x_l = self.mam(x_l)
        x_l, _ = self.attention2d(x_l)

        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)

        x_g = self.global_pool(x_g)
        x_g = x_g[:, :, 0, 0]

        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:, :, 0, 0]

        x_emb = self.neck(x_fused)
        return x_emb

    def freeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False

    def unfreeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = True
