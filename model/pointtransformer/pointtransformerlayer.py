import torch
import torch.nn as nn

from lib.pointops.functions import pointops


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x
    

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x
    
class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]



class PT(nn.Module):
    def __init__(self, block, cfg_mean, cfg_body, cfg_decoder, c=3, k=20):#, blocks, planes=[32, 64, 128, 256, 512], share_planes=8, stride=[1,4,4,4,4], nsample=[8,16,16,16,16], c=3, k=13):
        super().__init__()

        self._check_cfg_parameters(cfg_mean, "mean configuration")
        self._check_cfg_parameters(cfg_body, "body configuration")
        self._check_cfg_parameters(cfg_decoder, "decoder configuration")

        # Mean encoder parameters
        self.num_encoder_for_mean   = cfg_mean.num_encoder
        self.mean_planes            = cfg_mean.planes
        self.mean_blocks            = cfg_mean.blocks
        self.mean_share_planes      = cfg_mean.share_planes
        self.mean_stride            = cfg_mean.stride
        self.mean_nsample           = cfg_mean.nsample

        # Body encoder parameters
        self.num_encoder_for_body   = cfg_body.num_encoder
        self.body_planes            = cfg_body.planes
        self.body_blocks            = cfg_body.blocks
        self.body_share_planes      = cfg_body.share_planes
        self.body_stride            = cfg_body.stride
        self.body_nsample           = cfg_body.nsample

        # Decoder parameters
        self.num_decoder            = cfg_decoder.num_decoder
        self.decoder_planes         = cfg_decoder.planes
        self.decoder_blocks         = cfg_decoder.blocks
        self.decoder_share_planes   = cfg_decoder.share_planes
        self.decoder_nsample        = cfg_decoder.nsample

        # Mean Encoder
        self.c = c
        self.in_planes = c if c > 3 else 3
        self.mean_encoders = []
        for i in range(self.num_encoder_for_mean):
            self.mean_enc = self._make_enc(block, self.mean_planes[i], self.mean_blocks[i], self.mean_share_planes, stride=self.mean_stride[i], nsample=self.mean_nsample[i])  # N/(4**i)
            self.mean_encoders.append(self.mean_enc)
        self.mean_encoders = nn.ModuleList(self.mean_encoders)

        # Body Encoder
        self.c = c
        self.in_planes = c if c > 3 else 3
        self.body_encoders = []
        for i in range(self.num_encoder_for_body):
            self.body_enc = self._make_enc(block, self.body_planes[i], self.body_blocks[i], self.body_share_planes, stride=self.body_stride[i], nsample=self.body_nsample[i])  # N/(4**i)
            self.body_encoders.append(self.body_enc)
        self.body_encoders = nn.ModuleList(self.body_encoders)

        self.concat_planes = self.body_planes[-1] + self.mean_planes[-1]
        self.fusion_linear = nn.Sequential(nn.Linear(self.concat_planes, self.body_planes[-1]), nn.BatchNorm1d(self.body_planes[-1]), nn.ReLU(inplace=True))

        # The fusion layer outputs 'body_planes[-1]' channels (e.g., 256)
        self.in_planes = self.body_planes[-1]

        # Decoder
        self.decoders = []
        
        # We loop backwards: 4 -> 3 -> 2 -> 1
        # But we build the list forwards: Dec4, Dec3, Dec2, Dec1
        for i in range(self.num_decoder):
            # User condition: The first layer (i=0) is the head
            is_head = (i == 0)
            
            # We assume decoder_planes is [256, 128, 64, 32]
            plane = self.decoder_planes[i]
            
            self.dec = self._make_dec(
                block, 
                plane, 
                self.decoder_blocks[i], 
                self.decoder_share_planes, 
                nsample=self.decoder_nsample[i], 
                is_head=is_head
            )
            self.decoders.append(self.dec)
            
        self.decoders = nn.ModuleList(self.decoders)

        # Classification Head (matches the output of the last decoder)
        self.cls = nn.Sequential(
            nn.Linear(self.decoder_planes[-1], self.decoder_planes[-1]), 
            nn.BatchNorm1d(self.decoder_planes[-1]), 
            nn.ReLU(inplace=True), 
            nn.Linear(self.decoder_planes[-1], k)
        )
    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
    
    def _check_cfg_parameters(self, cfg, cfg_name="configuration"):
        num = cfg.num_decoder if "decoder" in cfg_name else cfg.num_encoder

        if len(cfg.planes) != num:
            raise ValueError(f"Length of planes list must be equal to num in {cfg_name}")
        if len(cfg.blocks) != num:
            raise ValueError(f"Length of blocks list must be equal to num in {cfg_name}")
        if len(cfg.nsample) != num:
            raise ValueError(f"Length of nsample list must be equal to num in {cfg_name}")
        if "decoder" not in cfg_name:
            if len(cfg.stride) != num:
                raise ValueError(f"Length of stride list must be equal to num in {cfg_name}")
        
    def forward(self, pxo, pxo_mean):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        p0_mean, x0_mean, o0_mean = pxo_mean
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        x0_mean = p0_mean if self.c == 3 else torch.cat((p0_mean, x0_mean), 1)

        # print("====== Start of Process ======")
        # print(f"Mean Tensor Shapes: {p0_mean.shape}, {x0_mean.shape}, {o0_mean.shape}")
        # print(f"Body Tensor Shapes: {p0.shape}, {x0.shape}, {o0.shape}")
        # print("==============================")

        # Mean Encoder Forward Pass
        # print("===== Mean Forward Pass ======")
        mean_features_p, mean_features_x, mean_features_o = [p0_mean], [x0_mean], [o0_mean]
        for i in range(self.num_encoder_for_mean):
            p0_mean, x0_mean, o0_mean = self.mean_encoders[i]([p0_mean, x0_mean, o0_mean])
            # print(f"Mean Tensor Shapes: {p0_mean.shape}, {x0_mean.shape}, {o0_mean.shape}")
            mean_features_p.append(p0_mean)
            mean_features_x.append(x0_mean)
            mean_features_o.append(o0_mean)
        # print("==============================")

        # Body Encoder Forward Pass
        # print("===== Body Forward Pass ======")
        features_p, features_x, features_o = [p0], [x0], [o0]
        for i in range(self.num_encoder_for_body):
            p0, x0, o0 = self.body_encoders[i]([p0, x0, o0])
            # print(f"Body Tensor Shapes: {p0.shape}, {x0.shape}, {o0.shape}")
            features_p.append(p0)
            features_x.append(x0)
            features_o.append(o0)
        # print("==============================")

        # Fusion at the bottleneck
        # print("===== Fusion Pass & Start of Decoder ======")
        fused_features = torch.cat((features_x[-1], mean_features_x[-1]), dim=1)
        # print(f"Fused Tensor Shape: {fused_features.shape}")
        x = self.fusion_linear(fused_features)
        p, o = mean_features_p[-1], mean_features_o[-1]
        # print(f"Tensor Shapes: {p.shape}, {x.shape}, {o.shape}")
        # print("==============================")

        # Decoder Forward Pass
        # print("===== Decoder Forward Pass ======")

        # Iterate through decoders
        # We need to pair the current bottleneck with the skip connection from the Mean Encoder
        # Skip connections are: mean_features_x[3], [2], [1], [0]
        decoder_layer = self.decoders[0]
        x = decoder_layer[1:]([p, decoder_layer[0]([p, x, o]), o])[1]
        
        for i in range(1, self.num_decoder):
            decoder_layer = self.decoders[i]
            
            # Get the skip connection target
            # If i=0, we need features from index -2 (the one before the bottleneck)
            skip_idx = -(i + 1) 
            
            p_skip = mean_features_p[skip_idx]
            x_skip = mean_features_x[skip_idx]
            o_skip = mean_features_o[skip_idx]
            # print(f"Skip Connection Tensor Shapes: {p_skip.shape}, {x_skip.shape}, {o_skip.shape}")
            # print(f"Tensor Shapes: {p.shape}, {x.shape}, {o.shape}")
            
            # 1. Transition Up (Interpolation + MLP)
            # decoder_layer[0] is TransitionUp
            up_x = decoder_layer[0]([p_skip, x_skip, o_skip], [p, x, o])
            
            # 2. Transformer Block
            # decoder_layer[1:] is the Sequence of Transformer Blocks
            # We pass the upsampled features + skip connection features (which TransitionUp handled internally usually, 
            # but here we pass the new coordinates p_skip)
            
            # Note: In standard PointTransformer, TransitionUp returns interpolated features.
            # We then feed that into the block.
            
            x = decoder_layer[1:]([p_skip, up_x, o_skip])[1]
            
            # Update current coordinates for next layer
            p = p_skip
            o = o_skip
        # print("==============================")

        # print("===== Final Forward Pass ======")
        x = self.cls(x)
        # print(f"Final Tensor Shapes: {x.shape}")
        # print("==============================")

        return x

def pt_repro(cfg_mean, cfg_body, cfg_decoder, **kwargs):
    model = PT(PointTransformerBlock, cfg_mean, cfg_body, cfg_decoder, **kwargs)
    return model