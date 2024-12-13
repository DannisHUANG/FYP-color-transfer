import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_transfer(source, target):
    # 将图像从 BGR 转换为 Lab 颜色空间
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 计算源图像和目标图像的均值和标准差
    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)

    # 对每个通道进行颜色匹配
    for i in range(3):  # L, a, b 通道
        source_lab[..., i] -= source_mean[i][0]
        source_lab[..., i] *= (target_std[i][0] / source_std[i][0])
        source_lab[..., i] += target_mean[i][0]

    # 将结果裁剪到 [0, 255] 范围
    source_lab = np.clip(source_lab, 0, 255)

    # 将图像从 Lab 转换回 BGR 颜色空间
    result = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return result

def plot_rgb_histograms(src, trg, transformed):
    # Plot histograms for RGB channels
    plt.figure(figsize=(18, 12))

    # R channel histograms
    plt.subplot(3, 3, 1)
    plt.title("Source - R channel")
    plt.hist(src[..., 0].ravel(), bins=256, color='red', alpha=0.5)
    plt.xlim([0, 255])

    plt.subplot(3, 3, 2)
    plt.title("Target - R channel")
    plt.hist(trg[..., 0].ravel(), bins=256, color='red', alpha=0.5)
    plt.xlim([0, 255])

    plt.subplot(3, 3, 3)
    plt.title("Transformed - R channel")
    plt.hist(transformed[..., 0].ravel(), bins=256, color='red', alpha=0.5)
    plt.xlim([0, 255])

    # G channel histograms
    plt.subplot(3, 3, 4)
    plt.title("Source - G channel")
    plt.hist(src[..., 1].ravel(), bins=256, color='green', alpha=0.5)
    plt.xlim([0, 255])

    plt.subplot(3, 3, 5)
    plt.title("Target - G channel")
    plt.hist(trg[..., 1].ravel(), bins=256, color='green', alpha=0.5)
    plt.xlim([0, 255])

    plt.subplot(3, 3, 6)
    plt.title("Transformed - G channel")
    plt.hist(transformed[..., 1].ravel(), bins=256, color='green', alpha=0.5)
    plt.xlim([0, 255])

    # B channel histograms
    plt.subplot(3, 3, 7)
    plt.title("Source - B channel")
    plt.hist(src[..., 2].ravel(), bins=256, color='blue', alpha=0.5)
    plt.xlim([0, 255])

    plt.subplot(3, 3, 8)
    plt.title("Target - B channel")
    plt.hist(trg[..., 2].ravel(), bins=256, color='blue', alpha=0.5)
    plt.xlim([0, 255])

    plt.subplot(3, 3, 9)
    plt.title("Transformed - B channel")
    plt.hist(transformed[..., 2].ravel(), bins=256, color='blue', alpha=0.5)
    plt.xlim([0, 255])

    plt.tight_layout()
    plt.show()

def main():
    # 读取源图像和目标图像
    source_img_path = r'C:\Users\UIC\Downloads\monai.jpg'  # 替换为你的源图像路径
    target_img_path = r'C:\Users\UIC\Downloads\fangao.jpg'  # 替换为你的目标图像路径

    source_img = cv2.imread(source_img_path)
    target_img = cv2.imread(target_img_path)

    if source_img is None:
        print("Error: Could not read the source image.")
        return

    if target_img is None:
        print("Error: Could not read the target image.")
        return

    # 执行颜色传输
    transferred_img = color_transfer(source_img, target_img)

    # 显示结果
    cv2.imshow('Source Image', source_img)
    cv2.imshow('Target Image', target_img)
    cv2.imshow('Transferred Image', transferred_img)

    # 绘制 RGB 直方图
    plot_rgb_histograms(source_img, target_img, transferred_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
