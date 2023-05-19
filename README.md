# Preparing Data

For this to work, you need to use GTSRB Data Preop.ipynb

You need to first unzip all the data you need

```
!unzip ./attack_data/Images_0_ppm.zip -d /content/data

!unzip ./attack_data/Images_1_ppm.zip -d /content/data

!unzip ./attack_data/Images_2_ppm.zip -d /content/data
```

These are all the extracted adversarial datasets of the three models.

There is also the natural dataset:

```
!unzip ../TRADES-with-German-Traffic-Signs/data/GTSRB_Final_Test_Images.zip -d  /content/data
```

Once you unzip the datasets, rememver to also manually drag GT-final_test.csv file into content/data/GTSRB/Final_Test/Images

Need to do the same for Images_0_ppm, etc.

These are the labels of the image data.