import os
import sys

# 确保当前目录在导入路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

class ImgNormalRatioNode:
    """图片比例"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "hd_width", "hd_height")
    FUNCTION = "generate"
    CATEGORY = "image"

    H_SCREEN = {
        "width": 832,
        "height": 472,
    }

    V_SCREEN = {
        "width": 472,
        "height": 832,
    }

    H_HD_SCREEN = {
        "width": 1920,
        "height": 1080,
    }

    V_HD_SCREEN = {
        "width": 1080,
        "height": 1920,
    }
    def __init__(self):
       pass

    def generate(self, ratio):
        if ratio == "16:9":
            return self.H_SCREEN["width"], self.H_SCREEN["height"], self.H_HD_SCREEN["width"], self.H_HD_SCREEN["height"]
        elif ratio == "9:16":
            return self.V_SCREEN["width"], self.V_SCREEN["height"], self.V_HD_SCREEN["width"], self.V_HD_SCREEN["height"]
