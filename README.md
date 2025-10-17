# Concurrence

This codebase provides the PyTorch-based software implementation for Concurrence, a novel method for discovering statistical dependencies between dynamical processess represented through a dataset of pairs of time series.

## 📦 Hardware Support

The software has currently been tested on the following OSs

* ✅ Linux
* ✅ MacOS

and the following hardware

* ✅ NVidia GPU (GeForce RTX 3090)
* ✅ CPU (AMD Ryzen Threadripper PRO)
* ✅ Apple Silicon (Apple M1 Pro)

👉 We **strongly suggest** the usage of an NVidia GPU, as it ran ~20x faster in our experiments.

## 🛠️ Installation

The code below will install the Concurrence package into a local environment via `pip`. While this code snippet uses `python3.11`, all versions of python listed below have been tested successfully, although one may need to change the versions of the packages in `requirements.txt` to ensure that the used python version is compatible with them:
* `python3.8`
* `python3.9`
* `python3.10`
* `python3.11`

The installation instructions below will create and activate a virtual environment.

```
git clone git@github.com:sariyanidi/Concurrence.git
cd Concurrence
python3.11 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🖥️ Demo
The command below runs the concurrence coefficient on a synthetic dataset of signal pairs `(x1,y1), (x2,y2), ..., (xN,yN)`.
```
python compute_concurrence.py data/xfiles.txt data/yfiles.txt \
                              data/xfiles_tes.txt data/yfiles_tes.txt --w=500
```
The data is expected to be organized as indicated in the files located within the first four command-line arguments (see also Documentation below).

The code by default runs on `cuda`, you can pass the argument `--device=cpu` for running it on CPU or `--device=mps` for running it on Apple Silicon.

## 🧪 Experiments on 100 synthetic datasets
You can reproduce the results in our experiments on the 100 synthetic datasets by first copying the datasets into your local Concurrence codebase. To this end, you must first download the synthetic datasets from the link below
* [https://drive.google.com/file/d/1oyhf_-uCaoj4ln5kPhjYmN1EQ_8_-pWX/view?usp=sharing](https://drive.google.com/file/d/1oyhf_-uCaoj4ln5kPhjYmN1EQ_8_-pWX/view?usp=sharing)
and unzip this file inside the `./data` directory, so that the unzipped directories are located as below.
```
./data/synth100/000
./data/synth100/001
...
./data/synth100/099
```
Then, you can run the experiments on all 100 datasets through the following code snippet
```console
for s in `seq 0 99`;
do
    ix=$(printf "%03d" $s);
    python compute_concurrence.py data/synth100/xfiles${ix}_tra.txt data/synth100/yfiles${ix}_tra.txt \
                                  data/synth100/xfiles${ix}_tes.txt data/synth100/yfiles${ix}_tes.txt \
                                  --w=500 --segs_per_pair=10;
done
```


## 📄 Documentation

### 💾 How to run with your own data? 

We created a simple **browser-based app** 🌐 that makes it easier for you to run conccurence on your data. Below we first describe how the concurrence code expects data to be organized, and then point to the browser-based app and its usage.

#### Training data
The code assumes that you have a dataset with signal pairs `(x1, y1), (x2, y2), ..., (xN, yN)`. Each of the signals `xi` is stored in a separate (text) file. Similarly, each `yi` signal is stored in a separate file. The `xi` signals are provided to the software with a `txt` file that contains their filepath. For example, you can create a file named `xfiles.txt` with a content like the one below.
```
/path/to/x1.txt 
/path/to/x2.txt 
...            
/path/to/xN.txt
```
Similarly, the you can create a file named `yfiles.txt` with a content like the one below.
```
/path/to/x1.txt 
/path/to/x2.txt 
...            
/path/to/xN.txt
```

Each of the time series `xi` must contain a time series of length `Ti` (i.e., a matrix of `1×Ti` or `Ti×1`). That is, variable-length time series are supported, although in our experiments so far the time series were approximately of the same length.

#### Testing data
The files `xfiles.txt` and `yfiles.txt` above will be used for training the network, and if you are using the code to compute the concurrence coefficient (as opposed to computing the PSCSs) you **must** have a separate pair of files that contain the test samples.  Suppose that we have two files called `xfiles_tes.txt` and `yfiles_tes.txt` that respectively contain the filepaths of the signals `xi` and the signals `yi` that will be used for testing. Then, you can compute the concurrence coefficient as below

```
python compute_concurrence.py xfiles.txt yfiles.txt xfiles_tes.txt yfiles_tes.txt --w=SEGMENT_SIZE
```

The parameter `--w` is the segment size (integer), and is the only parameter that you have to set. If you have no idea what the right value should be, try simply something like `T/3` where `T` is the average length of your signals `xi` and `yi`.


#### 🌐 Browser-based app

We created a Flask-based app that organizes the data and facilitates the running of the concurrence code. To run this app, simply type the command below to a terminal

```
python app.py
```

and then open a web browser and navigate to the address `http://127.0.0.1:8000/`.

You can use this app to ...

1. 🗂️ Organize your data, generate the command that runs concurrence and execute it in a separate terminal
1. 🧪 (Experimental) Organize your data and run concurrence on the web browser.

To see how to use this app, you can just run it on your browser. If you read the content above, running the app will hopefully be intuitive for you. 

### 🧬 How to run with multi-dimensional signals

The code readily supports multi-dimensional signals. You can follow the file structure laid out above, and put the (multi-dimensional) time series in the files `xi` and `yi`. It is assumed that each file `xi.txt` will contain `Ti×Kx` entries, where `Ti` is the temporal length of and `Kx` is the number of features (i.e., dimensions) in the time series `xi`. Similarly, the file `yi` must contain `Ti×Ky` entries, where `Ky` is the number of features in the time series `yi`.

### 🕒 How to produce the Per-segment Concurrence Scores (PSCSs)? 

The script `compute_concurrence.py` can also provide the PSCSs for the time series pairs. For this, the following arguments must be provided to the code
* `--xy_pair_ids_flist_traval` or `--xy_pair_ids_flist_tes`. (This parameter will be automatically handled for you if you use the 🌐 browser-based app above.)
  * If test data is provided (see Testing data above), then PSCSs will be computed on the test signals and `--xy_pair_ids_flist_tes` must be provided. This file must include as many rows as test signal pairs, and each row must have a unique ID corresponding to the signal pair.
  * Otherwise, PSCSs will be computed on the training data and `--xy_pair_ids_flist_traval` must be provided
* `--PSCS_file`, which is a path to a .csv file where the PSCS values will be written.

The `csv` file (`PSCS_file`) will store the PSCSs in a way that each row will contain all the PSCS values extracted from all segments (of size `w`) of a pair of sequences. The row will also contain an index that identifies the specific pair. For example, if your signals are stored in files such as `xsignal0001` and `ysignal0001`, the id will be `0001` (when you use the browser-based app). 

### ⚡ How to speed up the code? 

Reduce the number of `base_filters` or `segs_per_pair`. The `base_filters` controls the number of filters in the CNN that we used when computing concurrence (see also Online Methods section of the paper). The `segs_per_pair` controls the number of (concurrent/non-concurrent) segments that we automatically generate while computing the concurrence coefficient. The default values of both `base_filters` and `segs_per_pair` are likely overkill. You may significantly speed up by setting these to lower values.
