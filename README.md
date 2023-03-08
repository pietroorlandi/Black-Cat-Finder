# Cat Classification
### Reason of the project
The project is due to a personal need: _my black cat_ has an eating disorder and needs to eat only a more expensive type of food. However, I own several cats and would like this food to be dispensed only when the black cat is present.

### Goal
The goal of this project is to recognize if in the photo there is my black cat. In essence the project concerns about image classifcation one-vs-rest, where the positive class is the prensence of my black cat, and the negative is all the rest (other cat different the black, background, people, ...). </br>


### Section
The project initially is concerned with the creation of the recognition system, then when has recognized the black cat, it will give the food from the feeder.

### Data Collection
I have collected the photo through the _ESP32CAM_, the photo are 96x96 saved on JPEG format. The photos were taken personally trying to vary the angle of that photo a little bit and several days apart so as to vary the light conditions a little. Other changes will be introduced artificially with data-warping techniques.

<p align="center">
  <b> Examples of some photos </b></br>
<img src="https://github.com/pietroorlandi/Cat-Classification/blob/main/img/abbastanza_buone_mima%20(79).jpg" width="140">
<img src="https://github.com/pietroorlandi/Cat-Classification/blob/main/img/mimone_e_umani1%20(133).jpg" width="140">
<img src="https://github.com/pietroorlandi/Cat-Classification/blob/main/img/prova6%20(4).jpg" width="140">
<img src="https://github.com/pietroorlandi/Cat-Classification/blob/main/img/prova1_non_mima%20(13).jpg" width="140">
</p>

### Data Augmentation
Since that there was only 600 samples/photos for each class, it has been done data-augmentation technique to artificially increase the size of dataset. In particular, each new image can be randomly flipped, zoomed and rotated in a range of 30 degrees. <br>

