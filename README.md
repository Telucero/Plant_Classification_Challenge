<h1>2021 Data Science Kaggle Project </h1> 
<h3> Plant Classification </h3> 
<p> This is the group DataScience Kaggle Project for the Polytech' Nice Data Science course 2021. In this project we will be analyzing, constructing, and modeling Neural Networks to evaluate images for plant classification.</p>

<body>
    <b><h2> Members of Team Chikorita: </h2></b>  
    <ul>
        <li><h4> Ilaria Enache </h4></li>
        <li><h4> Lenny Klump </h4></li>
        <li><h4> Lynda Attouche </h4></li>
        <li><h4> Taylor Lucero </h4></li>
    </ul>

   <span style = "position:relative; left:350px; top:-130px"><img src="https://cdn.shopify.com/s/files/1/1034/3311/files/cymbidium-sp-pink-and-white-orchid-flowers-picture-id1093448542.jpg?v=1554867402" alt="Orchid" height=450px width=450px > </span>
       
</body>

## **Documentation**

The project is focused on plant classification. The dataset used is a subset of the public Pl@ntNet Dataset, it contains 140,256 images from 153 species.
We started by importing the required libraries for the project. We used pandas for data processing, torch for the deep learning framework, torchvision for the datasets, pre-trained models and manipulation functions, skimat for image preprocessing and seaborn for statistical graphs.

## **Exploration and analysis of the data**

For the data analysis we wanted to find statistics about our data.
We first loaded the data, either the class file or the images. We decided to create a dataframe that will contain them with their label and the "scientific" name of the label. We decided to do this because it was more practical for us to handle and it turned out to be the case. 

## **Preprocessing**

The data preprocessing has been applied to the whole dataset, to all images. 

**Transformation and data augmentation**

* Resizing Images: we resized our images from 600x600 to 224x224 to suit some models input as Resnet
* Normalization of images: so that all data are at the same scale which accelerates the training phase.
* Converting  images to tensor
* Data augmentation:to diversify the images, we have applied different techniques: RandomHorizontalFlip(), RandomRotation(50),..


**Splitting data** 

Since we do not have a valid test already defined in the data, we have divided our data set into two parts: a train set which will be used for training our model and a validation set which will be used for testing it when learning our neural network. To do this we chose to divide it as follows: 75% train set and 15% valid set.

**Resampling data**

During the analysis, we observed that the distribution of the images by classes was highly unbalanced (see graph...) which could affect our result in the end. 
So we decided to treat it in two different ways:
    
1) *Oversampling and undersampling:*
The goal here was to oversample the classes with a low number of images and undersample the ones with a lot.  To do so, we tested several sizes: 7000, 3000, 2000,1000,100
From these tests, sometimes, by oversampling we were easily confronted with an overfitting and by undersampling with an underfitting.

2) *Resampling using weightedRandomSampler:*
To apply this pytorch technique, we calculated the weight of each class and image. After that, the dataloader had to use the sample of classes with the highest weight. When testing this method, we obtained bad results, in particular for the accuracy. 

## **Model Experiments**

**Timeline** 

In order to maximize tests, we created two notebooks, where two persons tested independently. In the last week we chose the notebook which achieved a better accuracy and continued improving there. 

**Notebook 1:**

* First the LeNet5 Model was tested (similar to the Lab), but obviously gave very bad results due to enormous downsizing (600x600 to 28x28) and grayscale instead of RGB. 
* Afterwards AlexNet was tested, with various transformations (Resize, RandomFlip, Normalization). However the Architecture gave very bad results (val accuracy around 1% on first Epoch) 
* Then the world of pretrained models was discovered, and the VGG16 model was tested. Again various transformations were applied and NLLLoss and Adam optimizer were utilized. However the model again gave bad accuracy (val accuracy 5% on first epoch).
* After, we discovered that the dataset is extremely unbalanced so weighting the Loss function was tried but didn’t work out due to different class orders (coding mistake). 
* Then we continued trying to optimize the VGG16 model (different batch sizes, LossFunctions, optimizers,...) and finally got a validation accuracy of 56% with NLLLOss() and Adam optimizer. This was our best result so far. 
* After that, the ResNet architecture was tried: ResNet50, ResNet101 and ResNet152, where ResNet152 yielded the best results. Here, we also tried different Loss Functions and optimizers and finally achieved a validation accuracy of 72,5% with CrossEntropyLoss and Adam optimizer. This improved our best result.
* After that other things were tried: Weighted Loss, VGG19 architecture, EfficientNe_b7 architecture, using larger Images, Inception_v3 architecture. And all architectures with different Learning rates, batch sizes etc. However, nothing improved our current score. 

Because in Notebook 2 a scheduler was implemented, which seemed to work really well, the author of Notebook1 switched to work also on the code of Notebook2.

**Notebook2:**

We tested a couple of neural network models until we found the one that allows us to have the best score. We have in this case used pre-trained models.

* To start, 3 models of the resnets family have been applied to our data set. Indeed, these models have the potential to have a good performance on image classification. With deep layers, they allowed us to have good accuracy values. 
We have tested at the very beginning resnet50 whose architecture includes 50 deep layers. This one provided us with a good accuracy. However, to get more, looking at the amount of data we had, we thought that testing resnet101 and resnet152 would give a much better result. But it turns out that this was not the case.

* Another model known for its performance in image classification that we tested is the VGG-16. It allowed us to have the best accuracy at that time of the test and so we continued to work with it. Afterwards, another version was tested, the VGG19. But this one unfortunately could not provide a better score. 

* Moreover, we were curious to test another model, the efficientNet. A model which is based on a scaling method (scaling dimensions) that uses a set of fixed scaling coefficients. But this model did not converge to a better score.

* *Parameters tests*: In addition to testing different models, we have varied the parameters of our networks.
    * **Batch size**: We tested 4 values: 32, 64, 128, 256. In the end we ended up with 64

    * **Epochs**: It depended on the model, some of them performed in less epochs than others. But in general we ran our models on : 5, 10, 15 epochs. We wanted to test on more (which required more than 9 hours) but Kaggle did not allow that (max. 9h Runtime)..

    * **Loss function**: We tested and used in our final model crossEntropy() because it was the best adapted for a multi-class classification. However, we also tested NNLoss()

    * **Optimize**: We tested Stochastic Gradient Descent (SGD) with a momentum of 0.9 as well as Adam Optimizer. For the final result, we settled on the latter because it was the best. 

    * **Scheduler**: Having seen that the performance is not really increasing, we have introduced a new method for updating the learning rate: the scheduler. Indeed, this last one allows us to reduce the learning rate as the epochs increase. We have tested two types: CyclicLR which will be used for the final model and StepLR().


* During different tests various learning rates and batch sizes were tested on the pretrained VGG16 version with scheduler. Also, and most importantly, new transformations were applied. Using “RandomResizedCrop” instead of “Resize” and increasing the Epochs yielded us a new highest score: val_accuracy of 89,3% and when submitting a score of 82.6%. 

* Because of this high discrepancy between our validation score and the submission score we assumed that the test data isn’t following the same distribution as the Training data. So we revisited the concept of oversampling by manually balancing the images per class. Also, we applied even more transformations to increase generalizability: “ColorJitter”.  We ran our best scrong model with balanced data in two ways: 
    * 1. 2000 images per class and 6 epochs: This yielded  a validation accuracy of 76,5%. 
    * 2. 1000 imager per class and 12 epochs: This yielded a validation accuracy of 92%, which was a new best score. However, the submission score was only 74%.On the day of the deadline, we found out why the model didn’t perform well on submission: While balancing the classes we put a range(1,153) which only included 152 classes… A range (1,154) was needed. 
    
  **Visualisation**
  In the model, we made slight variations from the model worked on in the lab. These changes include gradient banks for both the training and validation sets, a iteration counter for both, and a gradient and store. Once the model begins training, the gradients calculated will then be appended after each epoch into these banks  for further use with seaborn. After this the information pulled from the model is converted into a Dataframe using a defined function where the data will be plotted in a series of code to produce visuals for the validation and training sets focusing on the gradients, accuracy, and loss. In these cases we want the gradient to be minimized, showing a plot decreasing and then plateuing. Accuracy to increase gradually after each epoch, showing that the model is learning. Loss to decrease gradually to show help calculate the gradients and show the models error during the training iterations. 


**Model Predictions** 

To predict the labels of the test data we created a loop and stored the results in a panda’s dataframe. To order the images we used a left join on the sample submission. 



