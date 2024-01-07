import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = (nn.Conv3d(in_ch, out_ch, (1,kernel_size,kernel_size), padding=(0,padding,padding), dilation=dilation, bias=False))
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))






class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag
        self.down_ = nn.Conv3d(in_ch, out_ch, (1, 4, 4), (1, 2, 2), (0, 1, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = einops.rearrange(x," b c t w h -> b (c t) w h ")
        if self.down_flag:
            x = self.down_(x)
        # x = einops.rearrange(x,"b (c t) w h -> b c t w h",t=4)

        return self.relu(self.conv(x))

# x = torch.zeros(size=(4,4,4,32,32))
# net = DownConvBNReLU(in_ch=4,out_ch=4)
# print(net(x).shape)


# def Upsample(dim):
#     return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag
        self.upsample = nn.ConvTranspose3d(in_ch//2, in_ch//2, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        # print(in_ch,out_ch)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            # print(x1.shape)
            x1 = self.upsample(x1)

        return self.relu(self.conv(torch.cat([x1, x2], dim=1)))





class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)
        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)
        return x + x_in
class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)
        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.en_1 = RSU(7,3,16,8)
        self.d1 = DownConvBNReLU(in_ch=8,out_ch=8)
        self.en_2 = RSU(6, 8, 16, 16)
        self.d2 = DownConvBNReLU(in_ch=16, out_ch=16)
        self.en_3 = RSU(5, 16, 16, 32)
        self.d3 = DownConvBNReLU(in_ch=32, out_ch=32)
        self.en_4 = RSU(4, 32, 16, 64)
        self.d4 = DownConvBNReLU(in_ch=64, out_ch=64)
        self.en_5 = RSU4F(64, 16, 64)
        self.d5 = DownConvBNReLU(in_ch=64, out_ch=32)
        self.en_6 = RSU4F(32, 16, 64)
        self.d6 = DownConvBNReLU(in_ch=64, out_ch=64)
    def forward(self,x):
        end = []
        x1 = self.en_1(x)
        end.append(x1)
        x1 = self.d1(x1)
        x2 = self.en_2(x1)
        end.append(x2)
        x2 = self.d2(x2)
        x3 = self.en_3(x2)
        end.append(x3)
        x3 = self.d3(x3)
        x4 = self.en_4(x3)
        end.append(x4)
        return end

