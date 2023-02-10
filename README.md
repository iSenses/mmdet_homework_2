# 作业2

## 作业要求

请参考 MMDetection 文档及教程，基于自定义数据集 balloon 训练实例分割模型，基于训练的模型在样例视频上完成color splash的效果制作，即使用模型对图像进行逐帧实例分割，并将气球以外的图像转换为灰度图像。
color splash样例： https://github.com/matterport/Mask_RCNN/blob/master/assets/balloon_color_splash.gif
注：由于GPU使用资源有限，请同学们尽量在CPU上先调通程序再进行模型的训练。
balloon是带有mask的气球数据集，其中训练集包含61张图片，验证集包含13张图片。
下载链接：https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip


## 作业汇报

![output.gif](https://github.com/iSenses/mmdet_homework_2/blob/main/output.gif)

模型使用了mask_rcnn_r50_fpn_1x_coco，暂时训练 40 epochs。

模型链接：https://pan.baidu.com/s/1iEp1m_VZ4-OwXxSafaRSnQ?pwd=h3yn 

- convert.py
  数据集转换为CocoDataset
- train.py
  训练
- output.py
  利用训练结果将test_video.mp4转化为color splash效果的output.mp4
  
- 一个脚本运行：run.sh
```bash
pip install -U pip
pip install torch-1.10.2-cp37-cp37m-manylinux1_x86_64.whl 
pip install -U torchvision==0.11.3 pycocotools openmim
mim install mmcv-full==1.5.0
mim install mmdet==2.22.0
mim download mmdet --config mask_rcnn_r50_fpn_1x_coco --dest .
curl -O https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip -q balloon_dataset.zip
sudo apt install -y ffmpeg

echo "convert.py==============================================================="
python convert.py
echo "train.py==============================================================="
mim train mmdet train.py
echo "output.py==============================================================="
python output.py
```

