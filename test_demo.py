import cv2
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import os
from lvae.utils import msssim as my_msssim
from lvae import get_model


def spectral_angle_mapper(spectrum1, spectrum2):
    """
    计算两个光谱向量之间的 Spectral Angle Mapping (SAM)

    参数:
        spectrum1: 第一个光谱向量
        spectrum2: 第二个光谱向量

    返回:
        sam: Spectral Angle Mapping
    """
    # 将光谱向量单位化
    spectrum1_normalized = spectrum1 / np.linalg.norm(spectrum1)
    spectrum2_normalized = spectrum2 / np.linalg.norm(spectrum2)

    # 计算两个向量之间的点积
    dot_product = np.dot(spectrum1_normalized, spectrum2_normalized)

    # 计算角度
    sam = np.arccos(dot_product)
    if np.isnan(sam):
        sam = np.pi/2

    # 将弧度转换为度
    sam = np.degrees(sam)

    return sam


def calculate_psnr_and_msssim(image_a_path, image_b_path):
    # 读取图像
    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)

    # 将图像转换为灰度图像（如果需要）
    gray_image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    # 计算PSNR
    psnr = peak_signal_noise_ratio(gray_image_a, gray_image_b)
    if np.isinf(psnr) or np.isnan(psnr):
        psnr = 60  #PSNR_MAX

    # 计算MS-SSIM
    ms_ssim = my_msssim.msssim(gray_image_a, gray_image_b)
    if np.isinf(ms_ssim) or np.isnan(ms_ssim):
        ms_ssim = 1.0  #SSIM_MAX

    return psnr, ms_ssim

def calculate_psnr_and_msssim_sam(image_a_path, image_b_path):
    # 读取图像
    image_a = np.load(image_a_path)
    image_a = image_a[10:40,:,:]
    image_b = np.load(image_b_path)
    C, H, W =image_a.shape[:]

    # 逐个波段计算PSNR和MS-SSIM
    psnr_all = []
    msssin_all = []
    for c in range(C):
        psnr_c = peak_signal_noise_ratio(image_a[c], image_b[c])
        if np.isinf(psnr_c) or np.isnan(psnr_c):
            psnr_c = 60  # PSNR_MAX
        ms_ssim_c = my_msssim.msssim(image_a[c], image_b[c])
        if np.isinf(ms_ssim_c) or np.isnan(ms_ssim_c):
            ms_ssim_c = 1.0  # SSIM_MAX
        psnr_all.append(psnr_c)
        msssin_all.append(ms_ssim_c)
    sam_all = []
    for i in range(H):
        for j in range(W):
            sam_all.append(spectral_angle_mapper(image_a[:,i,j], image_b[:,i, j]))
    psnr = sum(psnr_all)/C
    ms_ssim = sum(msssin_all)/C
    sam = sum(sam_all)/len(sam_all)

    return psnr, ms_ssim, sam


def compression(model, pretrained_path, image_path, compressed_path, lmb=2048):
    com_model = get_model(model, pretrained=pretrained_path)
    com_model.eval()
    com_model.compress_mode(True)

    com_model.compress_file(image_path, compressed_path, lmb=lmb)
    # 计算文件大小
    input_image = cv2.imread(image_path)
    height, width, channels = input_image.shape
    bits = 0
    if input_image.dtype == 'uint8':
        bits = 8
    elif input_image.dtype == 'uint16':
        bits = 16
    else:
        depth = input_image.dtype
        print(f"输入图像的位深不是8位也不是16位，而是{depth}")
    input_file_size = height * width * channels * bits / 8
    output_file_size = os.path.getsize(compressed_path)
    rate = input_file_size/output_file_size

    im = com_model.decompress_file(compressed_path)


    return rate, im


def compression_hsi(model, pretrained_path, image_path, compressed_path, lmb=2048, un_load=False):
    if not un_load:
        com_model = get_model(model, pretrained=pretrained_path)
        com_model.eval()
        com_model.compress_mode(True)
    else:
        com_model = un_load

    com_model.compress_file(image_path, compressed_path, lmb=lmb)
    # 计算文件大小
    input_image = np.load(image_path)
    imgjpg = np.transpose(input_image[[23, 13, 4], :, :], (1,2,0))
    cv2.imwrite(compressed_path.replace('.bin', '_org.jpg'), cv2.cvtColor(imgjpg, cv2.COLOR_RGB2BGR))
    channels, height, width = input_image.shape
    bits = 0
    if input_image.dtype == 'uint8':
        bits = 8
    elif input_image.dtype == 'uint16':
        bits = 16
    else:
        depth = input_image.dtype
        print(f"输入图像的位深不是8位也不是16位，而是{depth}")
    input_file_size = height * width * channels * bits / 8
    output_file_size = os.path.getsize(compressed_path)
    rate = input_file_size/output_file_size

    im = com_model.decompress_file(compressed_path)


    return rate, im


def PSNR_batch():
    # 图像路径
    list_dir = os.listdir('/home/ubuntu/data/HSI/test/val_hyspecnet11k/')
    model_path = '/home/ubuntu/data/qarv-release-main/runs/qarv_hsi/qarv_hsi_lower_lmb32_8192/best.pt'
    com_model = get_model('qarv_hsi_lower', pretrained=model_path)
    com_model.eval()
    com_model.compress_mode(True)
    all_psnr = []
    all_msssim = []
    all_sam = []
    all_rate = []
    for image_name in list_dir:
        # image_name = 'GF5B_AHSI_W77.0_N38.7_20220318_002805_L10000095374_VN_8bit_209.npy'
        image_path = f'/home/ubuntu/data/HSI/test/val_hyspecnet11k/{image_name}'
        compressed_path = '/home/ubuntu/data/qarv-release-main/results/OUT_hsi8bit.bin'
        output_path = '/home/ubuntu/data/qarv-release-main/results/output_hsi8bit.npy'

        lmb = 32
        bits = 8
        rate, im = compression_hsi('qarv_hsi_lower', pretrained_path=model_path, image_path=image_path,
                                   compressed_path=compressed_path, lmb=lmb, un_load=com_model)
        print(f'Compress rate is {rate}')
        #  im is a torch.Tensor of shape (1, 3, H, W), RGB, pixel values in [0, 1]
        if bits == 8:
            im_np = (im.squeeze(0) * 255).permute(1, 2, 0).numpy().astype('uint8')
        elif bits == 12:
            im_np = (im.squeeze(0) * 4096).permute(1, 2, 0).numpy().astype('uint16')
        elif bits == 16:
            im_np = (im.squeeze(0) * 65536).permute(1, 2, 0).numpy().astype('uint16')
        else:
            print("请确认保存图像的位深")

        # 保存为JPEG图像
        cv2.imwrite(output_path.replace('.npy', '.jpg'), cv2.cvtColor(im_np[:, :, [23, 13, 4]], cv2.COLOR_RGB2BGR))
        np.save(output_path, im_np.transpose(2, 0, 1))

        # 计算PSNR和MS-SSIM
        psnr_value, msssim_value, sam_value = calculate_psnr_and_msssim_sam(image_path, output_path)

        # 打印结果
        print("PSNR:", psnr_value)
        print("MS-SSIM:", msssim_value)
        print("SAM:", sam_value)
        all_psnr.append(psnr_value)
        all_msssim.append(msssim_value)
        all_sam.append(sam_value)
        all_rate.append(rate/5)
    print(f"average rate: {sum(all_rate)/len(all_rate)}")
    print(f"average bpp: {8/(sum(all_rate) / len(all_rate))}")
    print(f"average psnr: {sum(all_psnr) / len(all_psnr)}")
    print(f"average msssim: {sum(all_msssim) / len(all_msssim)}")
    print(f"average sam: {sum(all_sam) / len(all_sam)}")

if __name__ == '__main__':
    PSNR_batch()
    exit()
    # 图像路径
    image_name = 'GF5B_AHSI_W77.0_N38.7_20220318_002805_L10000095374_VN_8bit_209.npy'
    image_path =  f'/home/ubuntu/data/HSI/test/val_hsi128/{image_name}'
    compressed_path = '/home/ubuntu/data/qarv-release-main/results/OUT_hsi8bit.bin'
    output_path = '/home/ubuntu/data/qarv-release-main/results/output_hsi8bit.npy'
    model_path = '/home/ubuntu/data/qarv-release-main/runs/qarv_hsi/qarv_hsi_hysp11k_lmb32_8192_bppxiao/best.pt'
    lmb = 32
    bits = 8
    rate, im = compression_hsi('qarv_hsi_lower', pretrained_path=model_path, image_path=image_path, compressed_path=compressed_path, lmb=lmb)
    print(f'Compress rate is {rate}')
    #  im is a torch.Tensor of shape (1, 3, H, W), RGB, pixel values in [0, 1]
    if bits == 8:
        im_np = (im.squeeze(0) * 255).permute(1, 2, 0).numpy().astype('uint8')
    elif bits == 12:
        im_np = (im.squeeze(0) * 4096).permute(1, 2, 0).numpy().astype('uint16')
    elif bits == 16:
        im_np = (im.squeeze(0) * 65536).permute(1, 2, 0).numpy().astype('uint16')
    else:
        print("请确认保存图像的位深")

    # 保存为JPEG图像
    cv2.imwrite(output_path.replace('.npy','.jpg'), cv2.cvtColor(im_np[:,:, [23,13,4]], cv2.COLOR_RGB2BGR))
    np.save(output_path, im_np.transpose(2,0,1))

    # 计算PSNR和MS-SSIM
    psnr_value, msssim_value, sam_value = calculate_psnr_and_msssim_sam(image_path, output_path)

    # 打印结果
    print("PSNR:", psnr_value)
    print("MS-SSIM:", msssim_value)
    print("SAM:", sam_value)
