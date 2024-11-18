Here are the steps to train a handwriting generation model:
1.	Preprocess the data.  You can either preprocess and entire pre-picked dataset, or take a subset of a dataset and preprocess that. An example of a pre-picked dataset would be the PureEng dataset; here’s how you would preprocess this dataset:
```
python .\preprocess_data.py --data_root .\ pureEnglish\smoothed\ --max_len 916 --out_dir [datadir] --subset
```
Note: the max_len of 916 was picked here because that is the longest sequence length from the PureEng dataset. You should replace this with the maximum sequence length of your picked dataset, or you can replace this with a small number if you want to cap the sequence length to a shorter length.

To take a subset of a dataset, make sure the data is in a directory called “samples” and run the following commands:
```
python .\get_training_subset.py --num_train 60000 --num_val 10000
python .\preprocess_data.py --data_root .\samples\ --max_len [max_len] --out_dir [datadir] --subset
```
Note: the first command prints the sequence length distributions, including the maximum length for both the train and test sets. So, max_len in the second command should be determined by considering these statistics.

2.	Train the model and monitor training. To train a Generative RNN, run the following command:
```
python train.py --learning_rate 0.001 --optimizer adam --use_scheduler
```
To train a Style Equalization model:
```
python train.py --learning_rate 0.001 --optimizer adam --use_scheduler --style_equalization
```
Note: ideally, these commands should take data directory path as input, but the code currently has this path hard-coded. This should be amended immediately.

Run TensorBoard to see the loss curves as the model training progresses:
```
tensorboard --logdir=.
```
3.	Export the model. To export a Generative RNN:
```
python export.py --char_to_code_file [datadir]\char_to_code.pt --state_dict_file save\[best_epoch].pt
```
To export a Style Equalization model:
```
python export.py --char_to_code_file [datadir]\char_to_code.pt --state_dict_file save\[best_epoch].pt --style_equalization
```
The model is now saved at `hwg.onnx`