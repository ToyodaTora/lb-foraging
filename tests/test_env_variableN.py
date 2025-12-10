

#%% '''ラッパー等定義''' 
# @title
# Cell 2: ラッパー実装と利用例
import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple
from pettingzoo import ParallelEnv
from typing import Dict, Any, Tuple, List

from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete
import lbforaging
import lbforaging.foraging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#画面描画のための前処理
#描画用のライブラリをインポート
import numpy as np
import imageio
from IPython.display import HTML
from base64 import b64encode
from pyvirtualdisplay import Display

# import cv2

# display = Display(visible=0, size=(1400, 900))
# display.start()
# frames = []  # 画像フレームの取得用リスト



# @title
from PIL import Image, ImageDraw, ImageFont

class PheromoneWrapper(gym.Wrapper):
    def __init__(self, env, width, height):
        # Call the parent constructor, so we can access self.env later
        super(PheromoneWrapper, self).__init__(env)
        self.w, self.h = width, height
        self.players = env.unwrapped.players
        self.pher = PheromoneField(self.w, self.h)

        # フォント（必要なら truetype に差し替えてください）
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            self.font = ImageFont.load_default()

        # アルファ値を含む color_map (R,G,B,Alpha)
        self.color_map = {
            0: (0, 0, 0, 0),
            1: (0, 255, 0, 120),
            2: (0, 200, 0, 140),
            3: (0, 150, 0, 160),
            4: (255, 255, 0, 140),
            5: (255, 180, 0, 150),
            6: (255, 150, 0, 160),
            7: (255, 100, 0, 170),
            8: (255, 50, 0, 180),
            9: (255, 0, 0, 200),
        }

    def reset(self):
        self.pher = PheromoneField(self.w, self.h)
        obss, _ = self.env.reset()
        return obss, _

    def step(self, action):
        obss, rewards, dones, truncateds, info  = self.env.step(action)
        return obss, rewards, dones, truncateds, info


    def render(self):
        # 元のフレームを取得（numpy HxWx3）
        frame = self.env.render()
        base = Image.fromarray(frame).convert("RGBA")

        # 透過オーバーレイを作る（初期は完全透明）
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        pher = self.pher.map.cpu().numpy()  # shape: (H, W) with ints 0..9
        H, W = pher.shape

        # セル幅・高さ（整数）
        cell_w = base.width // W
        cell_h = base.height // H

        for y in range(H):
            for x in range(W):
                v = int(np.round(pher[x,y]))
                if v == 0:
                    continue

                x1 = x * cell_w
                y1 = y * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                color = self.color_map.get(v, (255, 0, 0, 150))
                # overlay に半透明矩形を描く（確実にアルファが残る）
                draw.rectangle([x1, y1, x2, y2], fill=color)

                # 数字を描く（文字は不透明で見やすく）
                text = str(v)
                # 文字サイズ取得して中央寄せ
                bbox = self.font.getbbox(text)   # (x0, y0, x1, y1)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x1 + (cell_w - tw) / 2
                ty = y1 + (cell_h - th) / 2
                # 白背景の小円や影を入れると見やすくなる（オプション）
                # ここでは黒文字の上に薄い白縁を入れる簡易的手法
                # 白縁（軽く）:
                outline = 1
                draw.text((tx-outline, ty), text, font=self.font, fill=(255,255,255,200))
                draw.text((tx+outline, ty), text, font=self.font, fill=(255,255,255,200))
                draw.text((tx, ty-outline), text, font=self.font, fill=(255,255,255,200))
                draw.text((tx, ty+outline), text, font=self.font, fill=(255,255,255,200))
                # 本文（黒）
                draw.text((tx, ty), text, font=self.font, fill=(0,0,0,255))

        # 合成（base の上に overlay を alpha_composite）
        # alpha_composite は両方 RGBA 必須かつ同サイズであること
        composed = Image.alpha_composite(base, overlay)

        return np.array(composed)

class PheromoneField:
    def __init__(self, width, height):
        self.map = torch.zeros(height, width, dtype=torch.float32)

    def get(self, x, y):
        return int(self.map[y, x].item())

    def set(self, x, y, value):
        self.map[y, x] = value

    # def decay(self, rate=0.01):
    #     self.map *= (1.0 - rate)



#%% '''テスト実行''' 

print("Hello World!!")

# %%
