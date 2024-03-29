# Preparing Data

For this to work, you need to use GTSRB Data Prep.ipynb

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

Once you unzip the datasets, remember to also manually drag GT-final_test.csv file into content/data/GTSRB/Final_Test/Images

Need to do the same for Images_0_ppm, etc.

These are the labels of the image data.

# Generate Noise

`noise_generation.py` reads all images from folders Images_resize (clean images) and Images_0, Images_1, and Images_2 (adversarial images) [backed up on google drive] to subtract them to visualize the perturbations. Need to manually replace the file names in the script to visualize the noise for specific models.

# Checkpoints

There are three checkpoints for the GTSRB dataset, including model_gtsrb_rn_adv1.pt, model_gtsrb_rn_adv6.pt, and model_gtsrb_rn_nat.pt, all saved within the checkpoints folder on drive.

# Grad CAM
`grad-cam-og.py` is the original grad cam script that is implemented from scratch and is not the most ideal. `grad-cam.py` utilizes an existing library for CAM and overlay the heat map on each image.