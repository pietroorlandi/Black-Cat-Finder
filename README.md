# Cat Classification

## Goal
The goal of this project is to recognize if in the photo there is my black cat. In essence the project concerns about image classifcation one-vs-rest, where the positive class is the prensence of my black cat, and the negative is all the rest (other cat different the black, background, people, ...). </br>
I have done this system that recognizes if the black cat is in the picture because black cat has a food problem and it should eat only a specific type (more expensive) of food. </br>

## Section
The project initially is concerned with the creation of the recognition system, then when has recognized the black cat, it will give the food from the feeder.

## Data Collection
I have collected the photo through the ESP32CAM, the photo are 96x96 saved on JPEG format.

<p align="center">
  <b> Examples of some photos </b></br>
<img src="https://github.com/pietroorlandi/Cat-Classification/blob/main/img/abbastanza_buone_mima%20(79).jpg" width="140">
<img src="https://github.com/pietroorlandi/Cat-Classification/blob/main/img/mimone_e_umani1%20(133).jpg" width="140">
<img src="https://github.com/pietroorlandi/Cat-Classification/blob/main/img/prova6%20(4).jpg" width="140">
<img src="https://github.com/pietroorlandi/Cat-Classification/blob/main/img/prova1_non_mima%20(13).jpg" width="140">
</p>

### Data Augmentation
Since that there was only 600 samples/photos for each class, it has been done data-augmentation through *ImageDataGenerator*. In particular, each new image can be randomly flipped, zoomed and rotated in a range of 30 degrees. <br>
Through data-augmentation we can increase the number of samples in the dataset by introducing some synthetic data
