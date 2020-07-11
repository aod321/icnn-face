# Face parsing via Interlinked Convolutional Neural Network(Pytorch reimplement)

---   
<div align="center">    
 
# Face Parsing via iCNN

[![Paper](ttps://arxiv.org/abs/1806.02479)
</div>
---

## Description   
This is a pytorch implementation of Zhou et al [(2015)](https://arxiv.org/abs/1806.02479).
NOTICE: We have released a upgraded version of iCNN naming STN-iCNN, check the [paper](https://arxiv.org/abs/2002.04831) or [code](https://github.com/aod321/stn-icnn) for more information.

The network archtecture is as following:
![image.png](https://i.loli.net/2020/07/11/uysz8nKw3VTAEpe.png)


## How to run

First, install dependencies

### clone project   
git clone https://github.com/aod321/icnn-face

### install requirements
cd icnn-face
pip install -r requirements.txt

### train stage1
python train_stage1.py

### train stage2
python train_stage2.py

all the checkpoints can be found at checkpoints_{uuid} or checkpoints_{parts_name}_{uuid}

### Results
![image.png](https://i.loli.net/2020/07/11/7uq3ZTU9aXGsfCc.png)

Comparison with State-of-the-art Methods on HELEN

![image.png](https://i.loli.net/2020/07/11/EcjnUa3GkdPDZSg.png)

Others
![image.png](https://i.loli.net/2020/07/11/9zJunZrc8EpbwNm.png)
