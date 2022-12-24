import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size = 1), 
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=1),
        nn.ReLU(inplace=True)        
    )
    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size =2, stride = 2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)
        
        self.up_trans_1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
        
        self.up_conv_1 = double_conv(1024, 512)
        
        self.up_trans_2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
        
        self.up_conv_2 = double_conv(512, 256)
        
        self.up_trans_3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
        
        self.up_conv_3 = double_conv(256, 128)
        
        self.up_trans_4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        
        self.up_conv_4 = double_conv(128, 64)
        
        self.out = nn.ConvTranspose2d(in_channels = 64, out_channels = 2, kernel_size = 1, stride = 1)
        
        
    def forward(self, image):
        # bs, c ,h, w
        # Encoder
        #print('image shape : ', image.shape)
        x1 = self.down_conv_1(image) #
        x2 = self.max_pool_2x2(x1)
        
        #print('x2 shape :', x2.shape)
        
        x3 = self.down_conv_2(x2) #
        x4 = self.max_pool_2x2(x3)
        #print('x4 shape :', x4.shape)
        
        x5 = self.down_conv_3(x4)#
        x6 = self.max_pool_2x2(x5)
        #print('x6 shape :', x6.shape)
        
        x7 = self.down_conv_4(x6)#
        x8 = self.max_pool_2x2(x7)
        #print('x8 shape :', x8.shape)
        
        x9 = self.down_conv_5(x8)
        #print('x9 shape :', x9.shape)
        
        x = self.up_trans_1(x9)
        x = self.up_conv_1(torch.cat([x, x7 ], 1))
        #print('x.shape : ', x.shape)
        
        x = self.up_trans_2(x)
        x = self.up_conv_2(torch.cat([x, x5 ], 1))
        #print('x.shape : ', x.shape)
        
        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x, x3], 1))
        #print('x.shape : ', x.shape)
        
        x = self.up_trans_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))
        #print('x.shape : ', x.shape)
        #x  = torch.squeeze(torch.sigmoid(self.out(x)), 1)
        x  = torch.softmax(self.out(x), 1)
        #x  = self.out(x)
        #print('Out : ', x.shape)
        #print('out unique', x.unique())
        #a=b
        return x

if __name__ == '__main__':
    image = torch.randn((1,3, 512, 512))
    model = UNet()
    out = model(image)
    #print('out[0][0] : ', out[0][0])
    #print('out[0][1] : ', out[0][1])
    print('done', out.shape)
    print('out unique', torch.unique(out))
    #var = torch.argmax(torch.squeeze(out), dim = 2)
    #print(var.shape, torch.unique(var))