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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class VarDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VarDoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.r1 = nn.ReLU(inplace=True)
        self.conv_mu = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_sigma = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.outconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.r2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.r1(x)
        mean = self.conv_mu(x)
        logvar = self.conv_sigma(x)

        z = torch.distributions.Normal(mean, torch.exp(logvar))

        x2 = self.outconv(z)
        x2 = self.bn2(x2)
        x2 = self.r2(x2)
        return x2, mean, logvar


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, bilinear=True, c=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64 * c)
        self.down1 = Down(64 * c, 128 * c)
        self.down2 = Down(128 * c, 256 * c)
        self.down3 = Down(256 * c, 512 * c)
        factor = 2 if bilinear else 1
        self.down4 = Down(512 * c, 1024 * c // factor)

        self.up1 = Up(1024 * c, 512 * c // factor, bilinear)
        self.up2 = Up(512 * c, 256 * c // factor, bilinear)
        self.up3 = Up(256 * c, 128 * c // factor, bilinear)
        self.up4 = Up(128 * c, 64 * c, bilinear)
        self.outc = OutConv(64 * c, n_classes)


        # self.att0 = SelfAttention(64*c)
        self.att1 = SelfAttention(128*c)
        self.att2 = SelfAttention(256*c)
        self.attLAT1 = SelfAttention(1024*c // factor)

    def forward(self, image, embedding):
        x = torch.cat([image, embedding], dim=1)
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x2 = self.att1(x2)

        x3 = self.down2(x2)
        x3 = self.att1(x3)

        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.attLAT1(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
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

