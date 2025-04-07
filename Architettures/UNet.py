import torch
import torch.nn as nn
from torch.optim import Adam
from torch import func as F

device = 'cpu'

def matrix_to_grid_4d(matrix, grid_rows, grid_cols):
    """
    Divide un tensore 4D in una griglia di sotto-matrici (patch) non sovrapposte.

    Args:
        matrix (torch.Tensor): tensore 4D da dividere, di forma [N, C, H, W].
        grid_rows (int): numero di patch lungo l'asse verticale (altezza).
        grid_cols
        (int): numero di patch lungo l'asse orizzontale (larghezza).

    Returns:
        torch.Tensor: tensore contenente le patch, di forma [N, grid_rows*grid_cols, C, patch_height, patch_width].
    """
    # Ottieni le dimensioni del tensore
    N, C, H, W = matrix.shape
    # Calcola le dimensioni di ogni patch
    patch_h = H // grid_rows
    patch_w = W // grid_cols

    # Usa unfold sugli assi spaziali (H e W)
    patches = matrix.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    # Ora patches ha shape: [N, C, grid_rows, grid_cols, patch_h, patch_w]
    # Riorganizza le dimensioni: unisci grid_rows e grid_cols in una singola dimensione
    patches = patches.contiguous().view(N, C, grid_rows * grid_cols, patch_h, patch_w)
    # Se preferisci avere la dimensione dei canali subito dopo quella delle patch:
    patches = patches.permute(0, 2, 1, 3, 4)  # Shape finale: [N, grid_rows*grid_cols, C, patch_h, patch_w]
    return patches


def grid_to_matrix_4d(patches, grid_rows, grid_cols):
    """
    Ricostruisce il tensore originale a partire da una griglia di patch.

    Args:
        patches (torch.Tensor): tensore contenente le patch, di forma
            [N, grid_rows*grid_cols, C, patch_h, patch_w].
        grid_rows (int): numero di patch lungo l'asse verticale.
        grid_cols (int): numero di patch lungo l'asse orizzontale.

    Returns:
        torch.Tensor: tensore ricostruito, di forma [N, C, grid_rows*patch_h, grid_cols*patch_w].
    """
    N, num_patches, C, patch_h, patch_w = patches.shape
    if num_patches != grid_rows * grid_cols:
        raise ValueError("Il numero totale di patch deve essere grid_rows * grid_cols")

    # Rimodella in [N, grid_rows, grid_cols, C, patch_h, patch_w]
    patches = patches.view(N, grid_rows, grid_cols, C, patch_h, patch_w)
    # Permuta le dimensioni per posizionare correttamente le patch:
    # da [N, grid_rows, grid_cols, C, patch_h, patch_w] a [N, C, grid_rows, patch_h, grid_cols, patch_w]
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    # Combina le dimensioni spaziali per ricostruire il tensore originale
    return patches.view(N, C, grid_rows * patch_h, grid_cols * patch_w)


class DetailerDenoiser(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.modules = [
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding='same'),
            nn.Tanh(),
        ]
        self.block = nn.Sequential(*self.modules)

    def forward(self, x):
        _in = x
        out = self.block(_in)
        return out + _in


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=1):
        """
        Parametri:
          - channels: numero di canali in input (e output)
          - num_heads: numero di teste per l'attenzione (default 1)
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)

        # Proiezioni 1x1 per Q, K e V
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)

        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        x: tensor di forma [B, C, H, W]
        """
        B, C, H, W = x.shape

        h = self.norm(x)

        q = self.q(h).reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = self.k(h).reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = self.v(h).reshape(B, self.num_heads, C // self.num_heads, H * W)

        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = torch.softmax(attn, dim=-1)

        # Applica l'attenzione su V
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)

        out = self.proj_out(out)
        return x + out


class Block2(nn.Module):
    def __init__(self, patch, imlp_act=True, c=8):
        super().__init__()
        self.imlp_act = imlp_act
        self.patch_dim = patch
        self.att = nn.MultiheadAttention(patch ** 2, 1, batch_first=True)

        self.convQ = nn.Conv2d(patch ** 2 * c, patch ** 2, 3, padding='same')
        self.convK = nn.Conv2d(patch ** 2 * c, patch ** 2, 3, padding='same')
        self.convV = nn.Conv2d(patch ** 2 * c, patch ** 2, 3, padding='same')

        self.convT = nn.Conv2d(patch ** 2 * c, patch ** 2, 3, padding='same')

        # self.imlp = IMLP(49 * 16, 256)  # 1024

    def forward(self, x):
        x, t = x
        # c1 = torch.stack((x, t), axis=1).view(x.shape[0], 2, 28, 28)
        patched = matrix_to_grid_4d(x, self.patch_dim, self.patch_dim)
        patched_time = matrix_to_grid_4d(t, self.patch_dim, self.patch_dim)

        patched = patched.flatten(1, 2)
        patched_time = patched_time.flatten(1, 2)
        # patched = patched.view(patched.shape[0], patched.shape[1] * patched.shape[2], patched.shape[3],
        #                       patched.shape[4])
        # patched_time = patched_time.view(patched_time.shape[0], patched_time.shape[1] * patched_time.shape[2],
        #                                 patched_time.shape[3], patched_time.shape[4])

        Q = torch.sigmoid(self.convQ(patched))
        K = torch.sigmoid(self.convK(patched))
        V = torch.sigmoid(self.convV(patched) + self.convT(patched_time))
        QFlatten = Q.flatten(2).permute(0, 2, 1)
        KFlatten = K.flatten(2).permute(0, 2, 1)
        VFlatten = V.flatten(2).permute(0, 2, 1)
        att, _ = self.att(QFlatten, KFlatten, VFlatten)

        att = att.permute(0, 2, 1).view(att.shape[0], att.shape[2], 1, patched.shape[3], patched.shape[3])

        out = grid_to_matrix_4d(att, self.patch_dim, self.patch_dim) + x
        return out


class VarDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VarDoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # è come padding same in questo caso
        self.r1 = nn.ReLU(inplace=True)

        self.cmu = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.cvar = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(1, out_channels, kernel_size=3, padding=1)
        self.r2 = nn.ReLU(inplace=True)

    def forward(self, x):
        _in = x
        x = self.r1(self.conv1(x))

        mean = self.cmu(x)
        logvar = self.cvar(x)

        epsilon = torch.randn_like(logvar)
        z = mean + torch.exp(logvar) * epsilon

        out = self.outconv(z)
        out = self.r2(out)

        return out, mean, logvar


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # è come padding same in questo caso
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # concatena lungo i canali
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, c=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Attention
        self.denoiser = DetailerDenoiser(c)
        self.att1 = SelfAttention(c * 2)
        self.att2 = SelfAttention(c * 4)
        self.attB = SelfAttention(c * 8)
        self.attB5 = SelfAttention(c * 16)

        # Encoder
        self.inc = DoubleConv(n_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)

        # Latent
        self.pool_latent = nn.AdaptiveAvgPool2d((16, 16))
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # Decoder
        self.up1 = Up(c * 16, c * 8, c * 8, bilinear)
        self.up2 = Up(c * 8, c * 4, c * 4, bilinear)
        self.up3 = Up(c * 4, c * 2, c * 2, bilinear)
        self.up4 = Up(c * 2, c, c, bilinear)
        self.outc = OutConv(c, n_classes)

        # Time Step Encoder
        self.time_inc = DoubleConv(1, c)
        self.time_down1 = Down(c, c * 2)
        self.time_down2 = Down(c * 2, c * 4)
        self.time_down3 = Down(c * 4, c * 8)

    def forward(self, x, t):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.att1(x2)
        x3 = self.down2(x2)
        x3 = self.att2(x3)
        x4 = self.down3(x3)

        # Time Encode
        t1 = self.time_inc(t)
        t2 = self.time_down1(t1)
        t3 = self.time_down2(t2)
        t4 = self.time_down3(t3)

        # Latent
        x4_latent = self.pool_latent(x4)
        x4_latent = self.attB(x4_latent)

        x5 = self.bottleneck(x4_latent)
        x5 = self.attB5(x5)

        # Decoder
        x = self.up1(x5, x4 + t4)
        x = self.up2(x, x3 + t3)
        x = self.up3(x, x2 + t2)
        x = self.up4(x, x1 + t1)
        x = self.denoiser(x)
        logits = torch.tanh(self.outc(x))
        return logits


model = UNet(3, 3, True, c=1)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()

optimizer = Adam(model.parameters(), lr=0.0001)
model = model.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
mean_losses = []
Att = SelfAttention(3)

mock = torch.randn(1, 3, 200, 200)
mock_time = torch.randn(1, 1, 200, 200)
out = model(mock, mock_time)

print(out.shape)

