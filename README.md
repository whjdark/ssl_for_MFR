# requirements
```
pip install -r requirements.txt
```

# dataset
https://aistudio.baidu.com/aistudio/datasetdetail/194177?lang=en

# training
```
python single_train.py --arch=MsvNetLite --num_train=1 --pretrain=pretrain\PPLCNet_x1_0_ssld_pretrained.pth --data_aug

python single_train.py --arch=MsvNetLite --num_train=1 --simsiam_pretrain=pretrain\MsvNetLite1.0_simsiam_pretrain_BS32_6000\checkpoint_0099.pth.tar --lr_sch=cos --data_aug --base_lr=30

python single_train.py --arch=MsvNetLite --num_train=1 --simsiam_pretrain=pretrain\MsvNetLite1.0_simsiam_pretrain_BS32_6000\checkpoint_0099.pth.tar --base_lr=30 --optim=sgdm --weight_decay=1e-6 --lr_sch=cos --freeze

python single_train.py --arch=FeatureNetLite --num_train=1 --simsiam_pretrain=pretrain\FeatureNetLite1.0_simsiam_pretrain_BS32_6000_40e\checkpoint_0039.pth.tar  --weight_decay=1e-4 --lr_sch=cos --data_aug --base_lr=20
```

# draw t-SNE

```
python single_train.py --arch=MsvNetLite --simsiam_pretrain=pretrain\MsvNetLite1.0_simsiam_pretrain_BS32_6000_200e\checkpoint_0199.pth.tar --program_type=draw_TSNE

python single_train.py --arch=MsvNetLite --program_type=draw_TSNE   

python single_train.py --arch=MsvNetLite --pretrain=pretrain\PPLCNet_x1_0_ssld_pretrained.pth --program_type=draw_TSNE   

python single_train.py --arch=MsvNetLite --model_path=output\2022_03_18_10_50_08\best_model.pth --program_type=draw_TSNE

# supervised 2022_03_09_16_31_21
```

# draw ROC and CM

```
####
# MsvNetLite
python single_train.py --arch=MsvNetLite --model_path=output\2022_03_17_12_00_03\best_model.pth --program_type=draw_ROC_CM

# FeatureNetLite

```
