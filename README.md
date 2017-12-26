# HandTrack-AutoEnc
Hand tracking with autoencoders

This is a module for hand tracking in video using autoencoders.

**Requirements**

This module was initialy written in these versions:

    Python 2.7
    OpenCV version 2

Most of the image loading/processing for training the model should also work in OpenCV version 3 (I've not tested this). However, the script to run the demo will need some changes, particularly in the parts where the video output is save to file.

**Usage**

There are 2 pre-trained models: bw_model.h5 (black and white input video) and col_model.h5 (colour input video).
The demo can be run using the file `htrack.py`. By default, the module uses the bw_model.h5, which takes a (224 x 224) image as input.
On the other hand, the color model uses a (192 x 192) image as input.

To run the demo with the color model:

    (1) Change the parameter COL in hds.py (set COL = True)
    (2) Change the parameter SIZE in hds.py (set SIZE = (192,192))
    (3) Change the model path in htrack.py (set model = load_model('col_model.h5'))
    (4) Run: python htrack.py

To save the output to file, run the following (for example, to save to output.avi):

    python htrack.py ./output.avi

The tracker basically runs every frame through the model and generates a binary mask with white points at every pixel where the prediction is above a threshold. By defualt, the threshold is set to to 0.3. To change it, set `thresh` in line 25, in `htrack.py`.
As a binary map is generated instead of a heatmap, this threshold heavily affects the otuput and may need adjusting.

**Dataset**

I used the hand tracking dataset from the Visual Geometry Group at Oxford University: http://www.robots.ox.ac.uk/~vgg/data/hands/ 
This contains 13050 images of hands, and bounding boxes indicating the position of the hands.
The script hds.py can be used to load the dataset:

    (1) Set RAW_DS_PATH = <Path to downloaded dataset>
    (2) import hds
        ...
        ds = hds.load_ds()
        x,y = map(np.array,zip(*ds))

This is used in line 52, 53 in `auto.py`.

For the pretrained models, I have used only the raw images along with their horizontal mirror image. The script contains parameters which allow inflating the dataset with other operations. To use these while loading the dataset, set the parameters in lines 20-31 in `hds.py`:

    (1) Vertical mirror image (set H_FLIP = True)
    (2) Horizontal mirror image (set V_FLIP = True)
    (3) Histogram equalization (set HIST_EQ = True)
    (4) Gaussian blur (set BLUR = -1 to not use this, set BLUR = (3,3) or the required kernel size)
    (5) Gamma correction (set GAMMA = [0.5,1.8], or any list of floats. To not use this, leave it an empty list)

**Training a new model**

This was mainly a quick exercise in training autoencoders and seeing its effectiveness for tracking without postprocessing. The pretrained models may not be the most general. To train a new model, run:

    python auto.py

This will load the dataset, train the model and save it in `auto.h5`.

**Notes on training for improvements**

(1) Ensure images are large: The images in the dataset are all bigger than (300 x 300). However, many of them have a thick black padding, with the actual image content covering a much smaller area. I tried training with the sizes set to (64 x 64), (128 x 128), (224 x 224). Using a large image size significantly helps the model (manually looking through the images, you can see that resizing the images to (128 x 128) reduces the hands to a few pixels in many images). I have not tried using images larger that (224 x 224).

(2) Trimming the model: I am pretty sure the model size can be reduced (though training would take longer). Overall, it is a standard autoencoder. The only thing to ensure is that max pooling layers are not used in the encoder (I've used strided convolutions instead).

(3) Normalization: I have not used any normalization while training. Code to normalize an image can be added in line 45 in `hds.py`.

**Demo**

A demo of the black and white model can be seen here: https://youtu.be/BrmC-h-ymPo
