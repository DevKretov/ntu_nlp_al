This is the instruction how to use our library.

Our library consists of four main components:
    - Dataset class
    - Model class
    - Trainer class
    - Strategy class

In order to use our library for Active Learning experiments, you need to specify the dataset you will train your model on. Please use configuration YAML files provided in config folder. There is one config.yaml folder which configures the whole library and also selects the dataset configuration file to use. It is found in config.yaml, run.selected_dataset. Please provide dataset configuration file without .yaml suffix.
There are two different examples how these config files can look like: conll2003_tagging.yaml, which is used for tagging task and which downloads the dataset from HuggingFace model hub found here: https://huggingface.co/datasets . The file news_classification.yaml shows how the locally stored dataset has to be configured. We hope the names of the parameters are pretty clear.

After configuring dataset, you need to configure the model. Currently we support all models supported by HuggingFace's transformers library. All the models have to be from Classification or TokenClassification class. The model can reside in the hub: https://huggingface.co/models or to be stored locally. The mechanism is the same as if you worked with transformers library. In the future we will support all Pytorch models.

After that, you have to configure the AL strategy. Strategies available for classification:
    - random
    - least_confidence
    - entropy
    - badge
    - kmeans
Strategies available for tagging:
    - random
    - least_confidence
In the configuration file, you can add many strategies and even repeat the same strategy several times if you wish. Simply enlarge the list of chosen strategies for that.

Each strategy is implemented in a separate file in strategies package found in our library. You can simply implement your own strategy. You have use _Strategy class as its parent and implement query() method which has to return indices of data points in unlabelled dataset.

We have integrated our library with Weights & Biases MLOps monitoring platform. If you don't wish to track models there, you can simply switch off W&B in config file: just set run.weights_and_biases_on parameter to False. If you want to use it in your own environment, provide you API key in reporting.weights_and_biases_key parameter. You can also decide if you want to store confusion matrices, artefacts, store predictions made on test set in every AL iteration, etc. Everything is configurable there, please read it through carefully. It is really large library in the end.

Finally, you have to initialise Trainer class. It is responsible for the whole process. You may find a Notebook file AL_Experiments.ipynb useful for seeing how it works. You can also go through main.py file, which is simply the same routine but not in a Notebook. Trainer class is fed with a Dataset instance, a Model instance, a configuration file and with parameters needed for training (like if there is the need for full training/how many AL iterations to run, how many epochs to train the model on during each AL iteration/batch sizes and many other).

You can also specify if you want to visualise the results of the AL experiment locally. We have implemented a robust way of visualisation of different metrics found in utils.py file. Visualisations are also configurable. They store the output of the run in the separate folder along with the best dev-set F1 metric model. We run dev-set evaluation after 5 epochs of training right before the testing. We do it in this way because of our limitedness in GPU resources, so we could not afford running dev-set evaluation after each training epoch. The best dev-set F1 metric model is the one which was capable of reaching the highest F1 score during all AL iterations, so that in the end of the experiment you receive the best model among all experiments.

The experimenting is also design in the way that utilises GPU. If you have one, just make sure you have model.use_gpu parameter set to True.

If you want to simply debug the pipeline, set app.debug_mode parameter to False.

Otherwise, make sure that you meet all criteria from requirements.txt file. You will need to have transformers, datasets, torch, seqeval, wandb, sklearn, tqdm.

We have tested the library in Colab. For Colab, please ensure you have everything from requirements.txt installed.

We also tested the library on the M1 Macbook Air. We provide requirements_m1.txt with M1 modules. Bear in mind that Tensorflow/Pytorch have to be installed via conda. Refer to these tutorials for that:

   For tensorflow: https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706
   For pytorch: https://betterprogramming.pub/how-to-install-pytorch-on-apple-m1-series-512b3ad9bc6


If you want to test it seemlessly, just run Colab notebook provided: AL_Experiments.ipynb.

The expected output is the finished run in Weights & Biases and locally saved best model among all those that we trained on every strategy you select. You also might want to visualise AL iterations locally, we have implemented simple plotting for comparison, but we strongly suggest using Weights&Biases, because they have implemented everything for us and they track everything for us. It's much more convenient to analyse the outputs there.