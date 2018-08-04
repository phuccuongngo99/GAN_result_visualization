# GAN_result_visualization


There are many renowned tools that greatly aid in visualizing results of different neural network models such as TensorBoard. These tools are especially useful if we need to compare quantifiable results such as mAP, precision, recall. However, TensorBoard, as of now, does not allows for the comparison of models' hyperparameter. Moreover, generative outputs in Generative Adversarial Network are often not quantifiable and are usually in form of gif (sequence of images) Thus, I wrote a small code to automate GAN hyperparameters tweaking and result visualisation.

Here are the features:
  - create_folder.py will take in config_template.json as input and create under a master_folder different folders, each of which containing different hyperparameters configurations stored in config.json file
  - excute_folder will execute training for each of the folders under master_folder
  - create_docs will gather the config.json and result.json in each folder under master_folder to write hyperparameter and result in a nice table of docx file.
