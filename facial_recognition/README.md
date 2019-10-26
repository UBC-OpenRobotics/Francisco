# Francisco

<b>face_recognition_dlib.py</b>
<ul>
  <li>createPersonData - use cam to extract images of person and attach a label</li>
  <li>createEncodings - uses pretrained model to extract 128d vector with facial encodings for database</li>
  <li>recognizeFaces - uses existing database to do facial recognition real-time (use GPU)</li>
</ul>

<b>facial_recon_CNN.py</b>
<ul>
  <li>createData - iterate over dataset of images and labeled folders, extracts and resizes images, samples testing data. prepares data for input to train CNN</li>
  <li>createModel -CNN with 3 Conv2D and 2 MaxPooling Layers</li>
  <li>trainModel - Uses model and training/testing data to train and save model. Plots accuracy and value accuracy vs epoch</li>
  <li>recognizeFace - Uses trained model to recognize faces in real time.</li>
</ul>


