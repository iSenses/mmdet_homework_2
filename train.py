_base_ = ['mask_rcnn_r50_fpn_1x_coco.py']
# _base_= ['mask_rcnn_r50_fpn_2x_coco.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

dataset_type = 'CocoDataset'
data_root = 'balloon'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train = dict(
        ann_file='balloon/annotations/train.json',
        img_prefix='balloon/train',
        classes=("balloon", )
     ),
    val = dict(
        ann_file='balloon/annotations/val.json',
        img_prefix='balloon/val',
        classes=("balloon", )
    ),
    test = dict(
        ann_file='balloon/annotations/val.json',
        img_prefix='balloon/val',
        classes=("balloon", )
    )
)

checkpoint_config = dict(interval=5)
#model = dict(bbox_head=dict(num_classes=1))
load_from = 'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
runner = dict(type='EpochBasedRunner', max_epochs=30)
optimizer = dict(lr=0.0008)
lr_config = dict(step=[10, 20])
evaluation = dict(interval = 2, metric=['bbox', 'segm'])
log_level =  'INFO'
resume_from = None
work_dir = 'res_me'
workflow = [('train', 1)]
