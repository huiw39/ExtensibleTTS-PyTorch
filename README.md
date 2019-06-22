# ExtensibleTTS-PyTorch
An extensible speech synthesis system, build with PyTorch and the original code is from r9y9's https://github.com/r9y9/nnmnkwii_gallery

## Quick Start

### Dependencies  
- python 3.6   
- CUDA 9.0
- pytorch     
- nnmnkwii   
- pyworld    
- pysptk    
- scipy    
- numpy    
- pickle

### Prepare Dataset    

*Note: the repo requires wav files with aligned HTS-style full-context lablel files.*

1. **Download a dataset**    

[cmu_slt_arctic](http://104.131.174.95/slt_arctic_full_data.zip).

2. **Unpack the dataset into `~/ExtensibleTTS-PyTorch/datasets`**    

   After unpacking, your tree should look like this for cmu_slt_arctic:
   ```
   ExtensibleTTS-PyTorch   
     |- datasets    
         |- slt_arctic_full_data
             |- label_phone_align
             |- label_state_align
             |- wav
             |- questions-radio_dnn_416.hed
   ```

### Training

1. **Preprocess the data to extract linguistic/duration/acoustic feature**
```
python preprocess.py
```

2. **Count min/max/mean/var/scale value of the data for input/output feature normalization**
```
python norm_params.py
```
3. **Train a model**
```python train.py --train_model duration
``` 
  * Use `--train_model acoustic` for training a acoustic model
  
4. **Label to speech waveform from a duration/acoustic checkpoint**
```
synthesis.py --duration_checkpint * --acoustic_checkpint *
```   

5. **Restore from a checkpoint**
```
train.py --restore_step *
```

# WIP  
- [ ] combined with [MTTS](https://github.com/Jackiexiao/MTTS) mandarin frontend.   
- [ ] batch inference for synthesis speedup.    
- [ ] scheduled sampling.    
- [ ] model pruning. 

# Reference       
- [nnmnkwii_gallery](https://github.com/r9y9/nnmnkwii_gallery)    
- [tacotron](https://github.com/keithito/tacotron)    
- [tacotron_pytorch](https://github.com/r9y9/tacotron_pytorch)    
