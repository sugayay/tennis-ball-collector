"""Inspection graphics made from actual captures and image-derived detections."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont


def system_font_path():
    configured = os.environ.get('PHYSICAL_AI_FONT')
    if configured:
        return configured
    paths = []
    windows_dir = os.environ.get('WINDIR') or os.environ.get('SystemRoot')
    if windows_dir:
        paths.append(Path(windows_dir)/'Fonts'/'meiryo.ttc')
    if os.name != 'nt':
        paths.extend(sorted(Path(os.sep).glob('mnt/*/Windows/Fonts/meiryo.ttc')))
    for path in paths:
        if path.is_file():
            return str(path)
    return 'meiryo.ttc'


def font(size):
    return ImageFont.truetype(system_font_path(),size)


def top_pixel(point):
    x,y,z = point
    focal = 500/math.tan(math.radians(24))
    return np.array([800+focal*x/(34-z),570-focal*y/(34-z)])


def save_visuals(output,config,fleet,fixed,report):
    canvas = Image.new('RGB',(1600,1080),'#10232c')
    canvas.paste(Image.open(output/'fleet_top.png'),(0,70))
    draw = ImageDraw.Draw(canvas)
    draw.text((40,18),'ロボ2台 ＋ 固定カメラ2台',font=font(32),fill='#f0f6f5')
    colors = {'A':'#aeea80','B':'#66dbe7'}
    for spec in fleet['robots']:
        x,y = top_pixel(spec['position_m'])
        color = colors[spec['id']]
        draw.ellipse((x-22,y-22,x+22,y+22),outline=color,width=3)
        lx,ly = x-54,y+35 if spec['id']=='A' else y-82
        draw.rounded_rectangle((lx,ly,lx+126,ly+44),radius=9,fill='#10232c',outline=color,width=2)
        draw.text((lx+9,ly+4),'ロボ '+spec['id'],font=font(25),fill=color)
    for spec in fixed['cameras']:
        x,y = top_pixel([spec['position_m'][0],spec['position_m'][1],0])
        direction = 1 if spec['id']=='A' else -1
        color = colors[spec['id']]
        draw.rounded_rectangle((x-15,y-11,x+15,y+11),radius=4,fill=color)
        draw.polygon([(x+direction*15,y-8),(x+direction*26,y-15),(x+direction*26,y+15),(x+direction*15,y+8)],fill=color)
        end = x+direction*115
        draw.line((x+direction*34,y,end,y),fill=color,width=4)
        draw.polygon([(end,y),(end-direction*14,y-9),(end-direction*14,y+9)],fill=color)
        lx = 22 if spec['id']=='A' else 1290
        draw.text((lx,y-86),'固定カメラ '+spec['id'],font=font(26),fill=color)
        draw.text((lx,y-45),'高さ10m / 仮配置',font=font(21),fill='#b9cbd0')
    for item in report['merged_ball_estimates']:
        x,y = top_pixel(item['position_m'])
        draw.ellipse((x-13,y-13,x+13,y+13),outline='#f8d17a',width=3)
        draw.text((x+16,y-24),str(item['ball_id']),font=font(24),fill='#f8d17a',
                  stroke_width=3,stroke_fill='#10232c')
    counts = [len(view['results']['uniform_tiles']['observations']) for view in report['views']]
    draw.text((40,963),f'固定A：{counts[0]}検出 / 固定B：{counts[1]}検出 / 重複をまとめて{len(report["merged_ball_estimates"])}球',font=font(28),fill='#aeea80')
    draw.text((40,1017),'球の印は画像から推定した位置 / 走行範囲は未決定 / 停止したシーンでの確認',font=font(23),fill='#b9cbd0')
    canvas.save(output/'fixed_camera_layout.png')
    for view in report['views']:
        camera_id = view['camera_id']
        raw = Image.open(output/f'camera_{camera_id}_rgb.png').convert('RGB')
        annotated = Image.open(output/f'camera_{camera_id}_yolo.jpg').convert('RGB')
        annotated.resize((1600,900),Image.Resampling.LANCZOS).save(output/f'camera_{camera_id}_preview.jpg',quality=92)
        observations = view['results']['uniform_tiles']['observations']
        if not observations:
            continue
        # The crop locations come from detections, not reference ball coordinates.
        ordered = sorted(observations,key=lambda item:item['bbox_xyxy'][2]-item['bbox_xyxy'][0])
        examples = [('小さく映った球',ordered[0]),('大きく映った球',ordered[-1])]
        strip = Image.new('RGB',(1000,510),'#10232c')
        drawing = ImageDraw.Draw(strip)
        for index,(label,item) in enumerate(examples):
            bounds = np.array(item['bbox_xyxy'])
            center = (bounds[:2]+bounds[2:])/2
            x,y = np.clip(np.round(center-64).astype(int),[0,0],np.array(raw.size)-128)
            patch = raw.crop((x,y,x+128,y+128)).resize((384,384),Image.Resampling.NEAREST)
            overlay = ImageDraw.Draw(patch)
            overlay.rectangle(tuple((bounds-np.array([x,y,x,y]))*3),outline='#aeea80',width=2)
            left = 44+index*500
            strip.paste(patch,(left,65))
            drawing.text((left,12),f'{label} / 球{item["ball_id"]}',font=font(26),fill='#f0f6f5')
            drawing.text((left,462),'元画像の128×128画素を3倍表示',font=font(22),fill='#b9cbd0')
        strip.save(output/f'camera_{camera_id}_details.png')
