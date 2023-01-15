# SiamPA

This project hosts the code for implementing the [SiamPA](https://www.worldscientific.com/doi/10.1142/S0219691323500054) algorithm for visual tracking. 

The raw results are [here](https://drive.google.com/file/d/1p0IbQUmGSpd1Mx_m-rphZq4jYK8wwIXu/view?usp=share_link). The code is based on [PySOT](https://github.com/STVIR/pysot) and [SiamBAN](https://github.com/hqucv/siamban).



## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using SiamPA

### Add SiamPAto your PYTHONPATH

```bash
export PYTHONPATH=/path/to/SiamPA:$PYTHONPATH
```

### Download models

Download models from [here]() and put the `model.pth` in the correct directory in experiments

### Webcam demo

```bash
python tools/demo.py \
    --config experiments/uav/config.yaml \
    --snapshot experiments/uav/model.pth
    # --video demo/bag.avi # (in case you don't have webcam)
```

### Download testing datasets

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [here](https://github.com/Giveupfree/SOTDrawRect/tree/main/SOT_eval). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker

```bash
cd experiments/uav
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset UAV123 	\ # dataset name
	--config config.yaml	  # config file
```

The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker

assume still in `experiments/uav`

``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset UAV123        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

###  Training :wrench:

See [TRAIN.md](TRAIN.md) for detailed instruction.

## License

This project is released under the [Apache 2.0 license](LICENSE). 
