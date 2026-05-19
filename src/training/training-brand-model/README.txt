Put these:
/models/car-brand/Vx_x/car_brand.pth
/models/car-brand/Vx_x/classes.pth

Structure needed:

training-brand-model/
│── dataset/
│   │── train/brand-1/images...
│   │── val/brand-1/images...
│
│── notebook.ipynb
│── README.txt

Then it creates these:

training-brand-model/
│── car_brand.pth
│── classes.pth

Move them to /models/(new_version)/

Modify line 24 in brand-predict.py file to the new version path

Finish...