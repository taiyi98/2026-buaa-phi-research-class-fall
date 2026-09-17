# HINT: Historical Interaction Transformer for Object Interaction Anticipation

本目录对应课题组"面向具身智能的第一人称视角交互行为理解与预测"项目中的第四项研究内容——**历史行为驱动的具身交互意图预测**，代码为对应 ICME 2026 投稿论文的实现。

## 任务简介

任务为 Ego4D 的 Short-term object interaction Anticipation (STA)：给定一段第一人称视频，预测接下来最可能发生的交互——物体类别（Noun）、动作类别（Verb）、物体位置（Bounding Box）、距接触发生的时间（Time-to-Contact, TTC）。

## 方法简介

在 StillFast（Ego4D STA 官方 baseline，[Ragusa et al., CVPR 2023]）的双分支架构基础上，提出 **HINT (Historical INteraction Transformer)**：

- 短期感知分支：处理临近帧的空间/时间特征（ResNet + RPN + RoIAlign，ViT-1B 视频backbone）
- 长期交互建模分支：将历史交互组织为时空语义图（动作节点/物体节点 + temporal edge/interaction edge），通过 **Graph-Infused Attention (GIA)** 融合标准自注意力与图结构注意力，提取用户长期行为意图的高层表征
- 两分支特征融合后送入多任务预测头，联合预测 Noun / Verb / BBox / TTC

在 Ego4D v1、v2 STA benchmark 上均达到 SOTA（v2: Overall mAP 7.23, N+V mAP 18.77；v1: Overall mAP 6.51, N+V mAP 16.98）。

## 目录结构

```
code/
├── main.py                  # 训练/验证/测试入口
├── configs/
│   ├── sta/                 # HINT/StillFast 在 Ego4D v1/v2 上的配置
│   └── simple_detection/    # Faster R-CNN 目标检测配置
├── stillfast/
│   ├── models/               # 模型定义（stillfast.py, faster_rcnn_sta.py,
│   │                          #   graph_transformer.py <- HINT/GIA核心实现, vit.py）
│   ├── datasets/             # Ego4D STA 数据集加载
│   ├── evaluation/           # mAP 等指标计算
│   ├── tasks/                # PyTorch Lightning Task 封装（STATask等）
│   └── ...
├── sg1_ckpt/
│   ├── config.yaml           # 一次训练实验使用的完整配置（供参考复现）
│   └── results/val.json      # 该次实验的验证集预测结果
├── remapped_nouns.json / remapped_verbs.json   # 类别映射表
└── requirements.txt
```

**注意：训练得到的模型 checkpoint（.ckpt）体积较大（数百MB~数GB），未上传到本仓库**，如需权重文件请联系课题组内部获取（存放于服务器 `/home/pty_ssd/Ego-understanding/ego_sta/sg1_ckpt/`）。

## 环境配置

```bash
conda create -n ego_sta python=3.9 -y
conda activate ego_sta

# torch 1.11.0 + cu113（V100/较新驱动均向下兼容cu113）
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

> 国内网络访问 GitHub（安装 detectron2 需要 `git clone`）经常超时，如遇 `Failed to connect to github.com` 之类的报错，配置好本地代理后重试：
> ```bash
> export http_proxy=http://127.0.0.1:7897
> export https_proxy=http://127.0.0.1:7897
> ```
> pip 默认走的 `pypi.ngc.nvidia.com` 索引在部分网络下不可达，可在 `pip.conf` 中去掉该索引，或改用国内镜像（如清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`）加速。

## 数据准备

数据集为 Ego4D STA v1/v2，需按官方流程申请下载（https://ego4d-data.org/ ，需签署许可协议，本仓库不重新分发数据）。下载后按以下结构组织，并在 config yaml 中对应修改路径：

```yaml
EGO4D_STA:
  ANNOTATION_DIR: /path/to/annotations      # fho_sta_{train,val,test_unannotated}.json
  FAST_LMDB_PATH: /path/to/lmdb             # 视频帧 lmdb
  STILL_FRAMES_PATH: /path/to/object_frames # 关键帧
```

## 使用方法

```bash
# 训练
python main.py --cfg configs/sta/STILL_FAST_R50_X3DM_EGO4D_v2.yaml --train

# 用某次实验目录里的 config + 最优 checkpoint 做验证
python main.py --test_dir sg1_ckpt --val

# 指定checkpoint测试
python main.py --cfg configs/sta/STILL_FAST_R50_X3DM_EGO4D_v2.yaml --test --checkpoint /path/to/xxx.ckpt
```

## 复现结果

当前 `sg1_ckpt` 为阶段性训练结果（epoch 0, 尚未完整训练），后续会补充完整复现的 Noun / N+TTC / N+V / Overall mAP 数据，对照论文 Table I（Ego4D v2）/ Table II（Ego4D v1）。

## 参考

- 本代码基于 [StillFast](https://github.com/fpv-iplab/StillFast)（Ragusa et al., CVPR 2023）二次开发
- Ego4D: Around the World in 3,000 Hours of Egocentric Video, CVPR 2022
