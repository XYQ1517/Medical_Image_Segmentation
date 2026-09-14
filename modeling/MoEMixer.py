import torch
import torch.nn as nn
from vmamba import vmamba_base_s2l15


# class EUM(nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_dim=None, scale=2,
#                  target_h=None, target_w=None):
#         super(EUM, self).__init__()
#         self.scale = scale
#         self.target_h = target_h
#         self.target_w = target_w
#
#         # 转置卷积单步上采样核心
#         # 使用 kernel_size=2*scale, stride=scale, padding=scale//2 确保空间尺寸精确 ×scale
#         self.trans_conv = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, in_channels,
#                                kernel_size=2 * scale, stride=scale, padding=scale // 2, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.GELU()
#         )
#
#         # 第一次上采样的通道投影: up_out_channels → out_channels (用于残差融合)
#         self.first_proj = nn.Conv2d(in_channels, out_channels,
#                                     kernel_size=1, stride=1, padding=0, bias=True)
#
#         # 最终通道投影: in_channels → hidden_dim (循环上采样后的最终统一)
#         self.pwc = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True)
#
#     def _upsample_trans_once(self, x):
#         """单次 ×scale 上采样 (转置卷积), 输出通道 = in_channels"""
#         return self.trans_conv(x)
#
#     def forward(self, x):
#         """
#         Args:
#             x: [B, C_in, H, W]
#         Returns:
#             first_up: 第一次上采样+通道投影的结果 [B, out_channels, H*scale, W*scale]
#             final:    循环上采样至目标尺寸后经pwc投影的最终特征 [B, hidden_dim, target_h, target_w]
#         """
#         raw_first_up = None
#         current = x
#
#         # 循环上采样直至达到目标空间尺寸
#         if self.target_h is not None and self.target_w is not None:
#             while current.shape[2] < self.target_h or current.shape[3] < self.target_w:
#                 current = self._upsample_trans_once(current)
#                 if raw_first_up is None:
#                     raw_first_up = current  # 原始第一次上采样结果 [B, in_channels, H*s, W*s]
#         else:
#             current = self._upsample_trans_once(current)
#             raw_first_up = current
#
#         # 第一次上采样的通道投影 → 用于与下一层残差融合
#         first_up = self.first_proj(raw_first_up)
#
#         # 最终通道投影至 hidden_dim
#         final = self.pwc(current)
#
#         return first_up, final


#   Efficient Upsample Module 
#   循环上采样至目标尺寸 + 通道统一，输出两个特征图
class EUM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=None, scale=2,
                 target_h=None, target_w=None):
        super(EUM, self).__init__()
        self.scale = scale
        self.target_h = target_h
        self.target_w = target_w

        # 单步上采样核心: depthwise conv + pixel shuffle实现 ×scale 上采样
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * scale * scale,
                      kernel_size=1, stride=1, padding=0, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels * scale * scale),
            nn.GELU()
        )

        # 第一次上采样的通道投影: up_out_channels → out_channels (用于残差融合)
        self.first_proj = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1, stride=1, padding=0, bias=True)

        # 最终通道投影: in_channels → hidden_dim (循环上采样后的最终统一)
        self.pwc = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True)

        # nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
        # nn.BatchNorm2d(hidden_dim // 2),
        # nn.ReLU(inplace=True),

    def _upsample_once(self, x):
        """单次 ×scale 上采样 (pixel shuffle), 输出通道 = in_channels / scale^2"""
        x = self.dwconv(x)
        B, C, H, W = x.size()
        # B, H, W, C
        x_permuted = x.permute(0, 2, 3, 1)
        # B, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((B, H, W * self.scale, int(C / self.scale)))
        # B, W*scale, H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # B, W*scale, H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view((B, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))
        # B, C/(scale**2), W*scale, H*scale
        x = x_permuted.permute(0, 3, 2, 1)
        return x

    def forward(self, x):
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            first_up: 第一次上采样+通道投影的结果 [B, out_channels, H*scale, W*scale]
            final:    循环上采样至目标尺寸后经pwc投影的最终特征 [B, hidden_dim, target_h, target_w]
        """
        raw_first_up = None
        current = x

        # 循环上采样直至达到目标空间尺寸
        if self.target_h is not None and self.target_w is not None:
            while current.shape[2] < self.target_h or current.shape[3] < self.target_w:
                current = self._upsample_once(current)
                if raw_first_up is None:
                    raw_first_up = current  # 原始第一次上采样结果 [B, up_out_ch, H*s, W*s]
        else:
            current = self._upsample_once(current)
            raw_first_up = current

        # 第一次上采样的通道投影 → 用于与下一层残差融合
        first_up = self.first_proj(raw_first_up)

        # 最终通道投影至 hidden_dim
        final = self.pwc(current)

        return first_up, final
    

#   Multi-head Mixier 
class MH_Mixer(nn.Module):
    """
      - 输入 [B, C, H, W]，将每个空间位置(h,w)视为一个token，共N=H×W个token
      - 通道维度C沿head方向分割为H个子空间
      - 每个head内部通过「全局聚合→广播」实现跨空间位置的token混合
      - 参数量远低于self-attention，且天然支持异构特征空间的混合
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

        # 每个head独立的子空间投影 (SplitHead后的子空间变换)
        self.head_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.head_dim, self.head_dim, 1),
                nn.BatchNorm2d(self.head_dim),
                nn.GELU(),
            ) for _ in range(num_heads)
        ])

        # 重建后的输出投影
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        shortcut = x

        # ===== Step 1: SplitHead (公式3): C → H × D_head =====
        x = x.reshape(B, self.num_heads, self.head_dim, H, W)     # [B, heads, D_head, H, W]

        # ===== Step 2: TokenMixing per head (公式4) =====
        mixed_heads = []
        for h in range(self.num_heads):
            feat_h = x[:, h]                                       # [B, D_head, H, W]

            # 全局token聚合: 对所有空间位置取全局统计 (等价于Concat后全局建模)
            global_ctx = feat_h.mean(dim=[2, 3], keepdim=True)     # [B, D_head, 1, 1]

            # 子空间投影
            proj_h = self.head_projs[h](feat_h)                    # [B, D_head, H, W]

            # 全局上下文注入 (广播到每个空间位置)
            mixed = proj_h + global_ctx * self.scale                # [B, D_head, H, W]
            mixed_heads.append(mixed)

        # ===== Step 3: Reconstruct (Concat各head结果) =====
        out = torch.cat(mixed_heads, dim=1)                         # [B, C, H, W]

        # ===== Step 4: Output projection =====
        out = self.proj(out)

        # ===== Step 5: Residual + LayerNorm =====
        B, C, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)  # [B, HW, C]
        out = self.norm(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        out = out + shortcut
        return out


#   multi-scale pixel decoder
class MS_Pixel_Decoder(nn.Module):
    def __init__(self, channels=[512,320,128,64], hidden_dim=256, num_heads=8):
        super(MS_Pixel_Decoder,self).__init__()
        self.hidden_dim = hidden_dim

        # # ===== 四层CSM (MH_Mixer): 并行处理backbone的4层特征 =====
        self.tm4 = MH_Mixer(dim=channels[0], num_heads=num_heads)
        self.tm3 = MH_Mixer(dim=channels[1], num_heads=num_heads)
        self.tm2 = MH_Mixer(dim=channels[2], num_heads=num_heads)
        self.tm1 = MH_Mixer(dim=channels[3], num_heads=num_heads)

        # ===== 三层EUM上采样: 循环上采样至统一尺寸 + 双级通道投影 =====
        # out_channels: 第一次上采样后的通道(用于残差融合)
        # hidden_dim:    最终循环上采样后的统一通道
        # F4^c → EUM: first_up(out=ch[1])与F3^c残差, final(hidden_dim)作为多尺度输出
        self.eum4_3 = EUM(in_channels=channels[0], out_channels=channels[1],
                          hidden_dim=hidden_dim, scale=2)
        self.eum3_2 = EUM(in_channels=channels[1], out_channels=channels[2],
                          hidden_dim=hidden_dim, scale=2)
        self.eum2_1 = EUM(in_channels=channels[2], out_channels=channels[3],
                          hidden_dim=hidden_dim, scale=2)

        # 最底层特征的直接通道投影 (无需上采样，已是最大尺寸)
        self.proj1 = nn.Conv2d(channels[3], hidden_dim, 3, 1, 1)

    def forward(self, x4, skips):
        """
        Args:
            x4: backbone最深层特征 [B, C4, H4, W4]
            skips: [x3, x2, x1] 浅层跳跃特征列表
        Returns:
            multi_scale_feats: List[Tensor], 4层统一通道数(hidden_dim)和统一空间尺寸的多尺度特征
        """
        x3, x2, x1 = skips

        # 确定目标统一空间尺寸 (以最底层f1_c为基准)
        with torch.no_grad():
            _tmp = x1
            target_h, target_w = _tmp.shape[2], _tmp.shape[3]

        # 动态设置各EUM的目标尺寸 (在推理前一次性设置即可)
        self.eum4_3.target_h = target_h
        self.eum4_3.target_w = target_w
        self.eum3_2.target_h = target_h
        self.eum3_2.target_w = target_w
        self.eum2_1.target_h = target_h
        self.eum2_1.target_w = target_w

        # ===== Step 1: 四层并行CSM (TokenMixer) =====
        f4_c = self.tm4(x4)   # [B, C4, H4, W4]  — 最小尺寸
        f3_c = self.tm3(x3)   # [B, C3, H3, W3]
        f2_c = self.tm2(x2)   # [B, C2, H2, W2]
        f1_c = self.tm1(x1)   # [B, C1, H1, W1]  — 最大尺寸

        # f4_c = x4   # [B, C4, H4, W4]  — 最小尺寸
        # f3_c = x3   # [B, C3, H3, W3]
        # f2_c = x2   # [B, C2, H2, W2]
        # f1_c = x1   # [B, C1, H1, W1]  — 最大尺寸

        # ===== Step 2: 逐层EUM循环上采样 + 通道投影 + 残差相加 (从高到低) =====
        # Layer 4 → 3: F4^c 循环上采样至统一尺寸, first_up用于与F3^c残差融合
        d4_first, d4_final = self.eum4_3(f4_c)       # d4_first: 中间×2结果, d4_final: 统一尺寸[B,D,H1,W1]
        f34 = d4_first + f3_c                          # [B, C3, ~H3*2, ~W3*2]

        # Layer 3 → 2: 融合结果循环上采样
        d3_first, d3_final = self.eum3_2(f34)
        f23 = d3_first + f2_c                          # [B, C2, ~H2*2, ~W2*2]

        # Layer 2 → 1: 融合结果循环上采样
        d2_first, d2_final = self.eum2_1(f23)
        f12 = d2_first + f1_c                          # [B, C1, ~H1*2, ~W1*2]

        # ===== Step 3: 输出4层多尺度特征 (全部统一为 hidden_dim × target_size) =====
        # 各EUM的final输出已包含: 循环上采样至target_size + pwc投影至hidden_dim
        # 最底层直接proj投影
        proj_f1 = self.proj1(f12)                     # [B, hidden_dim, H1, W1]

        # 4层多尺度特征: 各层的最终统一特征
        multi_scale_feats = [
            d4_final,   # 最高层特征 (经过完整循环上采样 + 通道统一)
            d3_final,   # 第3层融合后 (同上)
            d2_final,   # 第2层融合后 (同上)
            proj_f1,    # 最底层特征 (直接通道投影)
        ]

        return multi_scale_feats
    

#   Mixture-of-Experts Predictor
class MoEPredictor(nn.Module):
    """
    多尺度SwiGLU专家预测头
    - 每个多尺度特征对应一个独立SwiGLU专家，全部参与
    - 共享专家双分支:
        分支1 (gate): 生成各空间位置的门控权重, 与各专家输出相乘
        分支2 (expert): 生成全局特征, 与加权结果相加
    """
    def __init__(self, num_scales=4, hidden_dim=256, num_classes=1,
                 expert_inter_dim=None, shared_expert=True):
        super().__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.shared_expert = shared_expert

        if expert_inter_dim is None:
            expert_inter_dim = hidden_dim * 2

        # ========== 各尺度的SwiGLU专家 ==========
        # Expert_j(·) = FC_down(Swish(FC_gate(·)) ⊙ FC_up(·))
        self.expert_ups = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, expert_inter_dim, 1),
                nn.BatchNorm2d(expert_inter_dim),
            ) for _ in range(num_scales)
        ])
        self.expert_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, expert_inter_dim, 1),
                nn.SiLU(inplace=True),
            ) for _ in range(num_scales)
        ])
        self.expert_downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(expert_inter_dim, hidden_dim, 1),
                nn.ReLU(inplace=True),
            ) for _ in range(num_scales)
        ])

        # ========== 共享专家: 双分支设计 ==========
        self.fuse_conv = nn.Conv2d(hidden_dim * num_scales, hidden_dim, 1)

        # 分支1: 门控权重分支 → 输出 [B, num_scales, H, W] 的注意力权重
        self.shared_gate_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, num_scales, 1),
            nn.Softmax(dim=1),  # 在尺度维度归一化
        )

        # 分支2: 特征分支 → 输出 [B, D, H, W] 的残差特征
        self.shared_feat_up = nn.Conv2d(hidden_dim, expert_inter_dim, 1)
        self.shared_feat_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, expert_inter_dim, 1),
            nn.SiLU(inplace=True),
        )
        self.shared_feat_down = nn.Sequential(
            nn.Conv2d(expert_inter_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
        )

        # ========== 预测头: 两级转置卷积上采样至输入分辨率的4倍 ==========
        self.predictor = nn.Sequential(
            # 第一级: hidden_dim → hidden_dim//2, 上采样 ×2
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            # 第二级: hidden_dim//2 → num_classes, 上采样 ×2 (总计 ×4)
            nn.ConvTranspose2d(hidden_dim // 2, num_classes, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, multi_scale_feats):
        """
        Args:
            multi_scale_feats: List[Tensor], 长度=num_scales, 每个[B, D, H, W]
        Returns:
            pred: [B, num_classes, H, W]
        """
        concat_feats = torch.cat(multi_scale_feats, dim=1)          # [B, S*D, H, W]

        # ===== Step 1: 各尺度SwiGLU专家 =====
        expert_outputs = []
        for i in range(self.num_scales):
            feat = multi_scale_feats[i]                              # [B, D, H, W]
            up_out = self.expert_ups[i](feat)
            gate_out = self.expert_gates[i](feat)
            swiglu_out = up_out * gate_out                           # SwiGLU
            out = self.expert_downs[i](swiglu_out)                   # Expert_j(feat_j)
            expert_outputs.append(out)

        expert_stack = torch.stack(expert_outputs, dim=1)             # [B, S, D, H, W]

        # ===== Step 2: 共享专家双分支 =====
        fused = self.fuse_conv(concat_feats)                     # [B, D, H, W]

        # 分支1: 门控权重 → 对各专家输出加权
        gate_weights = self.shared_gate_branch(fused)             # [B, S, H, W], softmax归一化
        weighted_experts = expert_stack * gate_weights.unsqueeze(2)  # [B, S, D, H, W]
        expert_sum = weighted_experts.sum(dim=1)                  # Σ g_i · Expert_i

        # 分支2: 特征输出 → 作为残差相加
        sh_up = self.shared_feat_up(fused)
        sh_gate_val = self.shared_feat_gate(fused)
        shared_output = self.shared_feat_down(sh_up * sh_gate_val)  # SharedExpert

        output = expert_sum + shared_output                       # 加权专家 + 共享特征

        return torch.sigmoid(self.predictor(output))
    

class MoEMixer(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=92, num_heads=8):
        super(MoEMixer, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        path = "/home/wpf/SSD/XYQ/medical_image_segmentation/modeling/vssm_base_0229_ckpt_epoch_237.pth"
        self.backbone = vmamba_base_s2l15(pretrained=path)
        channels = [1024, 512, 256, 128]

        # pixel decoder initialization 
        self.pixel_decoder = MS_Pixel_Decoder(channels=channels, hidden_dim=hidden_dim,
                                          num_heads=num_heads)

        # 混合专家预测头 (SwiGLU多尺度专家)
        self.moe_head = MoEPredictor(
            num_scales=len(channels),
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            shared_expert=True
        )

        # # ========== 预测头: 两级转置卷积上采样至输入分辨率的4倍 ==========
        # self.predictor = nn.Sequential(
        #     # 第一级: hidden_dim → hidden_dim//2, 上采样 ×2
        #     nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(hidden_dim // 2),
        #     nn.ReLU(inplace=True),
        #     # 第二级: hidden_dim//2 → num_classes, 上采样 ×2 (总计 ×4)
        #     nn.ConvTranspose2d(hidden_dim // 2, num_classes, kernel_size=4, stride=2, padding=1),
        # )

    def forward(self, x):

        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)

        # encoder
        x1, x2, x3, x4 = self.backbone(x)

        # pixel decoder: 输出4层统一尺寸和通道的多尺度特征
        multi_scale_feats = self.pixel_decoder(x4, [x3, x2, x1])

        # MoE混合专家预测
        result = self.moe_head(multi_scale_feats)
        # result = self.predictor(multi_scale_feats[3])
        # return torch.sigmoid(result)
        return result

        
if __name__ == "__main__":
    from thop import profile
    from thop import clever_format
    x = torch.randn((1, 3, 256, 256)).to("cuda:0")
    model = MoEMixer().to("cuda:0")
    y = model(x)
    print(y.shape)
    MACs, Params = profile(model, inputs=(x,), verbose=False)
    Flops, Params = clever_format([MACs * 2, Params], '%.2f')
    print(f"Flops:{Flops}")
    print(f"Params:{Params}")
