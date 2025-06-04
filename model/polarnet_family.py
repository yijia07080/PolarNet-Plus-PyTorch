import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp
from functools import partial
from model.diffwire import dense_mincut_rewiring, dense_CT_rewiring, dense_mincut_pool
from torch.nn import Linear
from torch_geometric.nn import DenseGraphConv
import matplotlib.pyplot as plt
from torch.nn import init


class WeightCAM(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.feature = None
        self.gradient = None
        self.handlers = []
        self.handlers.append(module.register_forward_hook(self._get_features_hook))
        self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        """
        self.gradient = output_grad[0]

    def forward(self, target):
        # net.eval()
        # net.zero_grad()
        target.backward(retain_graph=True)
        # target.backward()
        gradient = self.gradient
        feature = self.feature

        if gradient is None:
            return torch.zeros((1, 8, 3), dtype=torch.float).cuda()

        B, _, C = gradient.shape
        gradient = gradient.permute(0, 2, 1)
        feature = feature.permute(0, 2, 1)

        gradient = gradient.reshape(B, 3, 8, 3, C)  #   torch.Size([1, 3 , 8 , 3, 72]
        feature = feature.reshape(B, 3, 8, 3, C)  #   torch.Size([1, 3 , 8 , 3, 72]

        gradient = torch.mean(gradient, dim=4)
        feature = torch.mean(feature, dim=4)

        cam = feature.cuda() * gradient.cuda()
        cam = torch.maximum(cam, torch.tensor([0.], dtype=torch.float).cuda())  # ReLU
        cam = F.adaptive_avg_pool3d(cam, (3, 8, 3))
        return cam.detach()[0, :, :]


class DiffWire(torch.nn.Module):

    def __init__(
        self,
        in_channels=64,
        out_channels=2,
        k_centers=192,
        derivative="normalized",
        hidden_channels=96,
        EPS=1e-15,
    ):
        super().__init__()

        self.EPS = EPS
        self.derivative = derivative

        # First X transformation
        self.lin1 = Linear(in_channels, hidden_channels)

        #Fiedler vector -- Pool previous to GAP-Layer
        self.pool_rw = Linear(hidden_channels, 2)

        #CT Embedding -- Pool previous to CT-Layer
        self.num_of_centers1 = k_centers  # k1 - order of number of nodes
        self.pool_ct = Linear(hidden_channels, self.num_of_centers1)  #CT

        #Conv1
        self.conv1 = DenseGraphConv(hidden_channels, hidden_channels)

        #MinCutPooling
        self.pool_mc = Linear(hidden_channels, 3 * 8 * 3)  #MC

        #Conv2
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)

        # MLPs towards out
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

        self.shape_avgpol = nn.AdaptiveAvgPool3d((3, 16, 4))

        self.adj_matrix = None

    def gen_adj_matrix(self, D, H, W):
        matrix = torch.zeros(D * H * W, D * H * W, dtype=torch.float)
        for x in range(D * H * W):
            matrix[x][x] = 1.

        node_vol = torch.tensor([x for x in range(D * H * W)], dtype=torch.int).reshape(D, H, W)

        for d in range(D):
            for h in range(H):
                for w in range(W):
                    node = node_vol[d, h, w]
                    if (node + 1) % W != 0:
                        matrix[node][node + 1] = 1.
                        matrix[node + 1][node] = 1.
                    if node % (H * W) not in range(H * W - W, H * W):
                        matrix[node][node + W] = 1.
                        matrix[node + W][node] = 1.
                    if node not in range(D * H * W - H * W, D * H * W):
                        matrix[node][node + H * W] = 1.
                        matrix[node + H * W][node] = 1.

        return matrix.unsqueeze(0)

    def get_adj_matrix(self, B, D, H, W, device):
        if self.adj_matrix is None:
            self.adj_matrix = self.gen_adj_matrix(D, H, W)

        adj = self.adj_matrix.clone().to(device)
        adj = adj.repeat(B, 1, 1)
        return adj

    def forward(self, x):

        # x [B, 64, 3, 134, 22] -> [B, Node, feature=64]

        x = self.shape_avgpol(x)  # -> x [B, 64, 3, 16, 4]

        x = x.permute(0, 2, 3, 4, 1)  # -> x [B, 3, 16, 4, 64]
        B, D, H, W, C = x.shape
        x = x.reshape(B, D * H * W, C)  # -> x [B, 3 * 16 * 4 = 192, 64]

        adj = self.get_adj_matrix(B, D, H, W, device=x.device)

        mask = torch.ones(B, D * H * W, dtype=torch.bool, device=x.device)

        x = self.lin1(x)

        #Gap Layer RW
        s0 = self.pool_rw(x)
        adj, mincut_loss_rw, ortho_loss_rw = dense_mincut_rewiring(x, adj, s0, mask, derivative=self.derivative, EPS=self.EPS, device=x.device)

        # print("dense_mincut_rewiring_x_Shape",x.shape) # x_Shape torch.Size([1, 192, 96])
        # print("dense_mincut_rewiring_adj_shape", adj.shape) # adj_shape torch.Size([1, 192, 192])
        # CT REWIRING
        # First mincut pool for computing Fiedler adn rewire
        s1 = self.pool_ct(x)
        adj, CT_loss, ortho_loss_ct = dense_CT_rewiring(x, adj, s1, mask, EPS=self.EPS)  # out: x torch.Size([20, N, F'=32]),  adj torch.Size([20, N, N])

        # print("dense_CT_rewiring_x_Shape", x.shape)  # x_Shape torch.Size([1, 192, 96])
        # print("dense_CT_rewiring_adj_shape", adj.shape)  # adj_shape torch.Size([1, 192, 192])

        # CONV1: Now on x and rewired adj:
        x = self.conv1(x, adj)
        # draw_adj_matrix((adj.detach() * 1e5).cpu()[0, :, :].numpy().astype(int))

        # MINCUT_POOL
        # MLP of k=16 outputs s
        s2 = self.pool_mc(x)
        # draw_adj_matrix((adj.detach() * 1e5).cpu()[0, :, :].numpy().astype(int))
        # Call to dense_cut_mincut_pool to get coarsened x, adj and the losses: k=16
        x, adj, mincut_loss, ortho_loss_mc = dense_mincut_pool(x, adj, s2, mask, EPS=self.EPS)  # out x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])
        # draw_adj_matrix((adj.detach() * 1e5).cpu()[0, :, :].numpy().astype(int))
        # print("dense_mincut_pool_x_Shape",x.shape) # [1, 72, 96]
        # print("dense_mincut_pool_adj_shape", adj.shape) # [1, 72, 72]
        # CONV2: Now on coarsened x and adj:
        x = self.conv2(x, adj)

        # Readout for each of the 20 graphs
        x = x.sum(dim=1)
        # Final MLP for graph classification: hidden channels = 32
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        main_loss = mincut_loss_rw + CT_loss + mincut_loss
        ortho_loss = ortho_loss_rw + ortho_loss_ct + ortho_loss_mc
        #ortho_loss_rw/2 + (1/self.num_of_centers1)*ortho_loss_ct + ortho_loss_mc/16

        return F.log_softmax(x, dim=-1), main_loss + ortho_loss, x, adj.detach()[0, :, :]


class PrintShape(nn.Module):

    def __init__(self, tag: str = "->", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tag = tag

    def forward(self, x):
        print(self.tag, x.shape)
        return x


class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNC
        else:
            x = x.permute(0, 2, 3, 4, 1)  # BCDHW -> BDHWC
        x = self.norm(x)
        return x


class UnPatchEmbed4dim(nn.Module):

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class UnPatchEmbed5dim(nn.Module):

    def forward(self, x):
        return x.permute(0, 4, 1, 2, 3)


class RNN3DBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = True,
        union="cat",
        with_fc=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2 * hidden_size if bidirectional else hidden_size
        self.union = union

        self.with_vertical = True
        self.with_horizontal = True
        self.with_3d = True
        self.with_fc = with_fc

        if with_fc:
            if union == "cat":
                self.fc = nn.Linear(3 * self.output_size, input_size)
            elif union == "add":
                self.fc = nn.Linear(self.output_size, input_size)
            elif union == "vertical":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_horizontal = False
            elif union == "horizontal":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_vertical = False
            else:
                raise ValueError("Unrecognized union: " + union)
        elif union == "cat":
            if 3 * self.output_size != input_size:
                raise ValueError(f"The output channel {2 * self.output_size} is different from the input channel {input_size}.")
        elif union == "add":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
        elif union == "vertical":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_horizontal = False
        elif union == "horizontal":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_vertical = False
        else:
            raise ValueError("Unrecognized union: " + union)

        self.rnn_v = RNNIdentity()
        self.rnn_h = RNNIdentity()
        self.rnn_d = RNNIdentity()

    def forward(self, x):
        # print("-> RNN3DBase ", x.shape)
        B, D, H, W, C = x.shape

        if self.with_horizontal:
            h = x.reshape(-1, W, C)
            h, _ = self.rnn_h(h)
            h = h.reshape(B, D, H, W, -1)
            # print("h", h.shape)

        if self.with_vertical:
            v = x.permute(0, 1, 3, 2, 4)
            v = v.reshape(-1, H, C)
            v, _ = self.rnn_v(v)
            v = v.reshape(B, D, W, H, -1)
            v = v.permute(0, 1, 3, 2, 4)
            # print("v", v.shape)

        if self.with_3d:
            d = x.permute(0, 3, 2, 1, 4)
            d = d.reshape(-1, D, C)
            d, _ = self.rnn_d(d)
            d = d.reshape(B, W, H, D, -1)
            d = d.permute(0, 3, 2, 1, 4)
            # print("d", d.shape)

        if self.with_vertical and self.with_horizontal:
            if self.union == "cat":
                x = torch.cat([v, h, d], dim=-1)
                # print("torch.cat([v, h, d], dim=-1)", x.shape)
            else:
                x = v + h + d
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h

        if self.with_fc:
            x = self.fc(x)
        # print("RNN3DBase -> ", x.shape)

        return x


class LSTM3D(RNN3DBase):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = True,
        union="cat",
        with_fc=True,
    ):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_3d:
            self.rnn_d = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class Sequencer3DBlock(nn.Module):

    def __init__(
            self,
            out_dim,
            hidden_size,
            img_size=(336, 56),
            patch_size=7,
            in_chans=64,
            embed_dim=192,
            mlp_ratio=3.0,
            rnn_layer=LSTM3D,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            num_layers=1,
            bidirectional=True,
            union="cat",
            with_fc=True,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        dim = out_dim
        channels_dim = int(mlp_ratio * dim)

        self.stem = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            flatten=False,
        )

        self.linear = nn.Linear(in_features=embed_dim, out_features=dim)

        self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional, union=union, with_fc=with_fc)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print("Sequencer3DBlock", x.shape)  # [1, 64, 336, 56]

        x = self.stem(x)
        # print("stem", x.shape)  # [1, 48, 8, 192]

        x = self.linear(x)
        # print("linear", x.shape)  # [1, 48, 8, 192]

        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        # print("rnn_tokens", x.shape)
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        # print("mlp_channels", x.shape)  # [1, 48, 8, dim]
        return x


class PolarBlock3D(nn.Module):

    def __init__(
        self,
        block_class,
        block_num,
        in_channels,
        out_channels,
        stride,
        dim,
        out_dim,
        hidden_size,
        img_size,
        patch_size,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cnn_3d = self._make_layer(block_class, out_channels, block_num, stride=stride)
        exp = block_class.expansion

        self.seq = Sequencer3DBlock(
            embed_dim=dim,
            out_dim=out_dim,
            hidden_size=hidden_size,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=out_channels * exp,
        )

        self.unpatchembd = UnPatchEmbed5dim()

        self.conv_last = nn.Conv3d(in_channels=out_dim, out_channels=out_channels, kernel_size=1)

        self.relu = nn.Sequential(nn.ReLU(), )

    def forward(self, x):
        x = self.cnn_3d(x)

        seq = self.unpatchembd(self.seq(x))

        seq = self.relu(seq)

        x = self.conv_last(seq)

        return x

    def _make_layer(self, block_class, out_channels, block_num, stride=1):
        downsample = None
        expanded_channels = out_channels * block_class.expansion
        if (stride != 1) or (self.in_channels != expanded_channels):
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, expanded_channels, kernel_size=1, stride=(stride, stride, stride), bias=False),
                nn.BatchNorm3d(expanded_channels),
            )

        layers = []
        layers.append(block_class(self.in_channels, out_channels, stride, downsample))
        for _ in range(1, block_num):
            layers.append(block_class(expanded_channels, out_channels))

        return nn.Sequential(*layers)


class ProjDetacher(nn.Module):

    def __init__(self, in_channels=1, out_channels: int = 64, down_sample: int = 10):
        super().__init__()

        self.dim = dim = 8
        self.channel_ext = nn.Sequential(
            # nn.LayerNorm(eps=1e-7),
            nn.Linear(in_features=3, out_features=dim),
            # nn.Linear(in_features=6, out_features=12),
            # nn.Linear(in_features=3, out_features=dim),
            Mlp(dim, dim * 3, act_layer=nn.GELU, drop=0.),
        )

        self.blk = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=False),
            # PrintShape(),
            nn.AdaptiveMaxPool3d((3, 224 * 6 // down_sample, 224 // down_sample)),
        )

    def forward(self, x):

        x = x.unsqueeze(1)  # B, 1, 3, H, W
        B, C, D, H, W = x.shape  # B, 1, 3, H, W
        x = x.permute(0, 1, 3, 4, 2)
        x = x.reshape(B, C * H * W, D)
        x = self.channel_ext(x)
        x = x.reshape(B, C, H, W, self.dim)
        x = x.permute(0, 1, 4, 2, 3)

        x = self.blk(x)

        return x


class GraphDiffWire(nn.Module):

    def __init__(
        self,
        k_centers=192,
        derivative="normalized",
        hidden_channels=96,
        EPS=1e-15,
    ):
        super().__init__()

        self.conv_fusion = DiffWire(
            k_centers=k_centers,
            derivative=derivative,
            hidden_channels=hidden_channels,
            EPS=EPS,
        )

    def forward(self, x):
        return self.conv_fusion(x)


class PolarNet3D(nn.Module):

    def __init__(
        self,
        k_centers=192,
        hidden_channels=96,
        EPS=1e-15,
    ):
        super().__init__()

        block_class = Bottleneck3D
        channel_base = 64
        down_sample = 10

        self.proj_d = ProjDetacher(out_channels=channel_base, down_sample=down_sample)
        block_num = [2, 3]
        in_channels = [channel_base * 1, channel_base * 1]
        out_channels = [channel_base * 1, channel_base * 1]
        stride = [1, 1]
        dim = [384, 384]
        hidden_size = [32, 32]
        img_size = [
            (1344 // down_sample, 224 // down_sample),
            (672 // down_sample // 2, 112 // down_sample // 2),
        ]
        patch_size = [1, 1]
        out_dim = [384, 384]

        self.stage_1 = PolarBlock3D(
            block_class=block_class,
            block_num=block_num[0],
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            stride=stride[0],
            dim=dim[0],
            out_dim=out_dim[0],
            hidden_size=hidden_size[0],
            img_size=img_size[0],
            patch_size=patch_size[0],
        )
        self.stage_2 = PolarBlock3D(
            block_class=block_class,
            block_num=block_num[1],
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            stride=stride[1],
            dim=dim[1],
            out_dim=out_dim[1],
            hidden_size=hidden_size[1],
            img_size=img_size[1],
            patch_size=patch_size[1],
        )

        self.graph_rewire = GraphDiffWire(
            k_centers=k_centers,
            hidden_channels=hidden_channels,
            EPS=EPS,
        )

        self.weight_cam = WeightCAM(self.graph_rewire.conv_fusion.conv2)

        self.weights_init_kaiming()

    def get_weight_map(self, x):
        self.zero_grad()
        return self.weight_cam(x[0][1])

    def weights_init_kaiming(self):
        classname = self.__class__.__name__
        if classname.find('Conv3d') != -1:
            init.kaiming_normal(self.weight.data)

    def forward(self, x, is_training=True):
        # [B, 1, 3, 1344, 224]
        # [B, C, D, H, W]

        # head_pro j torch.Size([1, 64, 8, 134, 22])
        # stage_1 torch.Size  ([1, 64, 8, 134, 22])
        # stage_2 torch.Size  ([1, 128,4, 67,  11])

        x = self.proj_d(x)
        # print("proj_d", x.shape)
        # [B, 64, 8, 134, 22]

        x = self.stage_1(x)
        # print("stage_1", x.shape)
        # [B, 64, 8, 134, 22]

        x = self.stage_2(x)
        # print("stage_2", x.shape)
        # [B, 128, 4, 67, 11]

        x, main_loss, _, adj = self.graph_rewire(x)

        adj = adj.cpu().numpy() if not is_training else None
        weight_cam = self.get_weight_map(x).cpu().numpy() if not is_training else None

        return x, main_loss, weight_cam, adj


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.stride = stride

        self.seq_1 = nn.Sequential(
            nn.Conv3d(
                inplanes,
                planes,
                kernel_size=(3, 3, 3),
                stride=(stride, stride, stride),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=False),
            nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(planes),
        )
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.seq_1(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.seq_1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=False),
            nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=False),
            nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * 4),
        )

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.seq_1(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RNNIdentity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(RNNIdentity, self).__init__()

    def forward(self, x):
        return x, None


def polarnet3d(k_centers=192, hidden_channels=96, eps=1e-15, **kwargs):
    return PolarNet3D(k_centers=k_centers, hidden_channels=hidden_channels, EPS=eps, **kwargs)