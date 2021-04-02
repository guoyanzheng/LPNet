import torch.nn as nn
import torch


class PixelShuffle3D(nn.Module):
    # partial borrowed from https://github.com/assassint2017/PixelShuffle3D/blob/master/PixelShuffle3D.py

    def __init__(self, scale_factor, is_reverse=False):
        """
        :param scale_factor(int,list,tuple): Scale up/down factor, if the input scale_factor is int,
         x,y,z axes of a data will scale up/down with the same scale factor,
         else x,y,z axes of a data will scale with different scale factor
        :param is_reverse(bool): True for HDC, False for DUC.
        """
        if isinstance(scale_factor, int):
            self.scale_factor_x = self.scale_factor_y = self.scale_factor_z = scale_factor
        elif isinstance(scale_factor, tuple) or isinstance(scale_factor, list):
            self.scale_factor_x = scale_factor[0]
            self.scale_factor_y = scale_factor[1]
            self.scale_factor_z = scale_factor[2]
        else:
            print("scale factor should be int or tuple or list")
            raise ValueError
        super(PixelShuffle3D, self).__init__()
        self.is_reverse = is_reverse

    def forward(self, inputs):
        batch_size, channels, in_depth, in_height, in_width = inputs.size()
        if self.is_reverse:  # for HDC
            out_channels = channels * self.scale_factor_x * self.scale_factor_y * self.scale_factor_z
            out_depth  = in_depth  // self.scale_factor_x
            out_height = in_height // self.scale_factor_y
            out_width  = in_width  // self.scale_factor_z
            input_view = inputs.contiguous().view(
                batch_size, channels,
                out_depth , self.scale_factor_x,
                out_height, self.scale_factor_y,
                out_width , self.scale_factor_z)
            shuffle_out = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
            return shuffle_out.view(batch_size, out_channels, out_depth, out_height, out_width)
        else:  # for DUC
            channels //= (
                        self.scale_factor_x * self.scale_factor_y * self.scale_factor_z)
            # out channels, it should equal to class number for segmentation task

            out_depth  = in_depth  * self.scale_factor_x
            out_height = in_height * self.scale_factor_y
            out_width  = in_width  * self.scale_factor_z

            input_view = inputs.contiguous().view(
                batch_size, channels,
                self.scale_factor_x, self.scale_factor_y, self.scale_factor_z,
                in_depth,            in_height,           in_width)

            shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
            return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)


class HDC(nn.Module):
    def __init__(self, downscale_factor):
        """
        reference paper: Zeng, G., & Zheng, G. (2019). Holistic decomposition convolution for effective semantic
         segmentation of medical volume images. Medical image analysis, 57, 149-164.

        3D HDC module, the input data dimensions should be 5D tensor like (batch, channel, x, y, z),
        :param downscale_factor(int, tuple, list): Scale down factor, if the input scale_factor is int,
         x,y,z axes of a data will scale down with the same scale factor,
         else x,y,z axes of a data will scale with different scale factor
        """

        super(HDC, self).__init__()
        self.ps = PixelShuffle3D(downscale_factor, is_reverse=True)

    def forward(self, x):
        x = self.ps(x)
        return x


class DUC(nn.Module):

    def __init__(self, upscale_factor, class_num, in_channels):
        """
        reference paper: Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R., ... & Wang, Z. (2016).
         Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network.
          In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1874-1883).

        3D DUC module, the input data dimensions should be 5D tensor like(batch, channel, x, y, z),
        workflow: conv->batchnorm->relu->pixelshuffle

        :param upscale_factor(int, tuple, list): Scale up factor, if the input scale_factor is int,
         x,y,z axes of a data will scale up with the same scale factor,
         else x,y,z axes of a data will scale with different scale factor
        :param class_num(int): the number of total classes (background and instance)
        :param in_channels(int): the number of input channel
        """
        super(DUC, self).__init__()
        if isinstance(upscale_factor, int):
            scale_factor_x = scale_factor_y = scale_factor_z = upscale_factor
        elif isinstance(upscale_factor, tuple) or isinstance(upscale_factor, list):
            scale_factor_x = upscale_factor[0]
            scale_factor_y = upscale_factor[1]
            scale_factor_z = upscale_factor[2]
        else:
            print("scale factor should be int or tuple")
            raise ValueError
        self.conv = nn.Conv3d(in_channels, class_num * scale_factor_x * scale_factor_y * scale_factor_z, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(class_num * scale_factor_x * scale_factor_y * scale_factor_z)
        self.relu = nn.ReLU(inplace=True)
        self.ps = PixelShuffle3D(upscale_factor, is_reverse=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.ps(x)
        return x


if __name__ == "__main__":
    ## test HDC's pixel shuffle and DUC's pixel shuffle modules by the ordered sample.
    scale_factor = (4,2,2)
    test_in = torch.arange(512).reshape((1, 1, 8, 8, 8))
    hdc_ps = PixelShuffle3D(scale_factor, is_reverse=True)
    duc_ps = PixelShuffle3D(scale_factor,
                            is_reverse=False)  # remember to add conv->batchnorm->relu before DUC pixel shuffle
    test_hdc_ps = hdc_ps(test_in)
    test_duc_ps = duc_ps(test_hdc_ps)
    error = test_duc_ps - test_in
    print("in data:\n", test_in)
    print("after hdc:\n", test_hdc_ps)
    print("after duc:\n", test_duc_ps)
    print("diff tensor:\n", error)
    print("\nsum:\n", error.sum())
    print("\nshape of init data:\n", test_in.shape)
    print("\nshape of hdc output:\n", test_hdc_ps.shape)
    print("\nshape of recover data:\n", test_duc_ps.shape)

    ## test HDC-3D UNet
    from model import UNet3D_HDC
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = UNet3D_HDC(16, 5, final_sigmoid=False, f_maps=16, number_of_fmaps=4, num_groups=4, scale_factor=scale_factor)

    test_model = test_model.to(device=device)
    test_to_model = torch.normal(0.0, 1.0, size=(2,1,256,256,256), device=device)
    test_out_model = test_model(test_to_model)
    print("\nshape of output data:\n", test_out_model.shape)
