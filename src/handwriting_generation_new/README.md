More complete readme coming soon

For best results repro, first:
```
python train.py --learning_rate 0.0001 --use_scheduler --num_epochs 80
```

Then, with the [path] as the path to best model between 70 and 80 epochs:
```
python train.py --resume_from_checkpoint [path] --learning_rate 0.0001
```