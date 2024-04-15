# Creating a new neural network

A typical workflow will consist of the following steps:


## 1. Getting the data ready (Getting the CustomDataset)

1. Create an instance of the `Labels` class from the nn_preprocessor.py file.
   - You can add your own `get_label()` method to this class to satisfy your experiment needs. 

2. Create an instance of `CustomDataset` class and pass the `data_dir` and labels method (from `Labels` class)

3. Choose a transform that fits your needs (if any), or create your own. The end result must be a `torch.Tensor` object.
   - If you decide on using ColPadding(), you will need to pass the `all_cols` attribute from the `CustomDataset` instance to the transform.
   - You can use `CustomToTensor()` to convert the `CustomDataset` object to a `torch.Tensor` object.

4. Create a `transforms.Compose([...])` that includes your transform(s).

5. Call `init_transforms()` from the `CustomDataset` instance and pass the `transform.Compose(...)` to it to initialize the transform.


## 2. Optional: Split the data into train, validation, and test sets

Using the `DatasetSplitter` class, you pass the `CustomDataset` instance and the `train_ratio` and `test_ratio` parameters to the constructor. The `train_ratio` and `test_ratio` parameters are the percentages of the data that will be used for training and testing, respectively. The remaining data will be used for validation (if any).


## 3. Create a data loader

This process is similar to the one described in the PyTorch documentation.


## 4. Create a neural network
You will need to create:

1. A `Hyperparameters` class that will hold the hyperparameters for your experiment.
2. A neural network `nn.Module` class that will hold the architecture of your neural network.
3. A criterion (loss) function.
4. optionally, an optimizer.
5. optionally, a scheduler.

Then pass all to the `net_train()` function from the `nn_modules.py` file.
