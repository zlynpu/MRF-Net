import torch
import torch.nn as nn
import torch.nn.functional as F


class ResContextBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResContextBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=(1,1))
        self.act1 = nn.LeakyReLU(True)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding=1)
        self.act2 = nn.LeakyReLU(True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding=2,dilation=2)
        self.act3 = nn.LeakyReLU(True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):

        shortcut = self.act1(self.conv1(x))
        x = self.bn1(self.act2(self.conv2(shortcut)))
        x = self.bn2(self.act3(self.conv3(shortcut)))

        out = x + shortcut
        return out



class Block1Res(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True, return_skip=True):
        super(Block1Res, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.return_skip = return_skip
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)
        resA = shortcut + resA1

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)
            if self.return_skip:
                return resB, resA
            else:
                return resB
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB
        
class Block2(nn.Sequential):
    def __init__(self,dropout_rate=0.2,kernel_size=3,pooling=True,drop_out=True):
        module = [
            nn.Dropout2d(p=dropout_rate) if drop_out else nn.Identity(),
            nn.AvgPool2d(kernel_size=kernel_size,stride=2,padding=1) if pooling else nn.Identity(),
        ]
        super(Block2, self).__init__(*module)


class Block4(nn.Module):
    def __init__(self, in_channels, out_channels,skip_channels,upscale_factor=2,dropout_rate=0.2,drop_out=True):
        super(Block4, self).__init__()

        self.upscale = nn.PixelShuffle(upscale_factor=upscale_factor)

        self.conv1 = nn.Conv2d((in_channels + skip_channels)//(upscale_factor**2), out_channels, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU(True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU(True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU(True)
        self.bn3 = nn.BatchNorm2d(out_channels)


        self.conv4 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU(True)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(p=dropout_rate) if drop_out else nn.Identity()

    def forward(self,x,skip):
        
        upcat = torch.concat([x,skip],dim=1)
        upcat = self.dropout(upcat)

        infeat = self.upscale(upcat)
        infeat = self.dropout(infeat)

        cat1 = self.bn1(self.act1(self.conv1(infeat)))
        cat2 = self.bn2(self.act2(self.conv2(cat1)))
        cat3 = self.bn3(self.act3(self.conv3(cat2)))

        cat = torch.concat([cat1,cat2,cat3],dim=1)

        out = self.bn4(self.act4(self.conv4(cat)))
        out =self.dropout(out)

        return out
    
class Block_withoutskip(nn.Module):
    def __init__(self, in_channels, out_channels,upscale_factor=2,dropout_rate=0.2,drop_out=True):
        super(Block_withoutskip, self).__init__()

        self.upscale = nn.PixelShuffle(upscale_factor=upscale_factor)

        self.conv1 = nn.Conv2d(in_channels//(upscale_factor**2), out_channels, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU(True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU(True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU(True)
        self.bn3 = nn.BatchNorm2d(out_channels)


        self.conv4 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU(True)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(p=dropout_rate) if drop_out else nn.Identity()

    def forward(self,x):

        x = self.upscale(x)
        x = self.dropout(x)

        # upcat = torch.concat([x,skip],dim=1)
        # upcat = self.dropout(upcat)


        cat1 = self.bn1(self.act1(self.conv1(x)))
        cat2 = self.bn2(self.act2(self.conv2(cat1)))
        cat3 = self.bn3(self.act3(self.conv3(cat2)))

        cat = torch.concat([cat1,cat2,cat3],dim=1)

        out = self.bn4(self.act4(self.conv4(cat)))
        out =self.dropout(out)

        return out
    
class UpBlock(nn.Module):

    def __init__(self, in_filters, out_filters, dropout_rate=0.2, drop_out=True, mid_filters=None):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.mid_filters = mid_filters if mid_filters else in_filters // 4 + 2 * out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(self.mid_filters, out_filters, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = upE1
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE
    
class conv_skip(nn.Module):
    def __init__(self, in_filters, out_filters, mid_filters):
        super(conv_skip, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.mid_filters = in_filters + mid_filters

        self.conv1 = nn.Conv2d(self.mid_filters, self.out_filters, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(self.out_filters)

    def forward(self, x, skip):
        upA = torch.cat((x, skip), dim=1)
        upE = self.conv1(upA)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)
        upE = upE1

        return upE

class UpBlock_withoutskip(nn.Module):

    def __init__(self, in_filters, out_filters, dropout_rate=0.2, drop_out=True):
        super(UpBlock_withoutskip, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters
        # self.mid_filters = mid_filters if mid_filters else in_filters // 4 + 2 * out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        # self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(self.in_filters // 4, out_filters, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        # upB = torch.cat((upA, skip), dim=1)
        # if self.drop_out:
            # upB = self.dropout2(upB)

        upE = self.conv1(upA)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = upE1
        if self.drop_out:
            upE = self.dropout2(upE)

        return upE