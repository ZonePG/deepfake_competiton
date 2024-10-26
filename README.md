# DeepFake Competition

## 采帧和提取人脸

将下好的数据集的中文路径名字剔除，数据集路径组织文件格式如下：
```
pre_id/
    0001/
        0001.jpg
        0002.jpg
        ...
pre_test/
    0001.mp4
    0002.mp4
    ...
train/
    0001.mp4
    0002.mp4
    ...
```

处理视频的脚本如下：
```
python3 annotation/process_video.py \
    --preprocess_path 数据集路径 \
    --interval 采帧的间隔秒数，例如 1 \
    --img_size 所裁后的人脸 resize 之后的大小，例如 224 \
    --backend 人脸检测模型，例如 yolov8 \
```

以 1s 间隔采帧为例，人脸大小为 224，使用 yolov8 作为人脸检测模型为例子，运行脚本后，会在数据集路径下生成采帧后的原始图片和提取的人脸图片，文件夹结构如下：
```
frames_1s/
    pre_test/
        0001/
            0001.jpg
            ...
    train/
        0001/
            0001.jpg
            
faces_1s_yolov8_224/
    pre_test/
        0001/
            0001.jpg
            ...
    train/
        0001/
            0001.jpg
            ...
```

## 数据集划分

5 折交叉验证划分后的数据文件已经在 annotation 文件夹下，下面脚本可以不用再手动运行。数据集划分脚本如下：
```
cd annotation
python3 preprocess_split_train_val.py 
```

## 训练和验证

训练脚本如下：
```
python3 main.py --video_path 数据集路径，例如 xxx/faces_1s_yolov8_224/train
```

注意这里要带 train 路径，train 路径下包含了训练集和验证集，训练过程中会根据 annotation 文件夹下的 txt 文件进行划分训练集和验证集。

## 测试集预测（TODO）
