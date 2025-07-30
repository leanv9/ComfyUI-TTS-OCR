import os
import sys

# 确保当前目录在导入路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

class ImgRatioNode:
    """图片比例"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 768, "min": 16, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
            },
            "optional": {
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("new_width", "new_height", "new_hd_width", "new_hd_height")
    FUNCTION = "generate"
    CATEGORY = "image"

    NORMAL_RATIO = 768*512
    NORMAL_HD_RATIO = 1920*1080

    def __init__(self):
       pass

    def generate(self, width: int, height: int):
        new_width, new_height = self._generate(width, height, self.NORMAL_RATIO)
        new_hd_width, new_hd_height = self._generate(width, height, self.NORMAL_HD_RATIO)
        return new_width, new_height, new_hd_width, new_hd_height

    def _generate(self, width: int, height: int, target_resolution: dict):
        """
        等比例缩放原始宽度和高度到目标分辨率范围内，返回最接近的宽度和高度。

        参数:
        width (int): 原始宽度
        height (int): 原始高度
        target_resolution (dict): 目标分辨率

        返回:
        tuple: 最接近的宽度和高度
        """
        if width <= 0 or height <= 0 or target_resolution <= 0:
            raise ValueError("宽度、高度和目标分辨率必须为正整数")
        
        # 计算原始宽高比
        aspect_ratio = width / height
        # 根据宽高比和目标分辨率计算新的宽度和高度
        # 设新宽度为x，新高度为y，x/y = aspect_ratio，x*y = target_resolution
        # 则 x = sqrt(target_resolution * aspect_ratio)，y = sqrt(target_resolution / aspect_ratio)
        new_width = int((target_resolution * aspect_ratio) ** 0.5)
        new_height = int((target_resolution / aspect_ratio) ** 0.5)
        # 调整宽度和高度为8的倍数
        new_width = max(8, round(new_width / 8) * 8)
        new_height = max(8, round(new_height / 8) * 8)

        return new_width, new_height
