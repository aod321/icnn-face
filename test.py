from dataset import SinglePart,TestStage1, TestStage2, HelenDataset
import torch
import os
from torchvision import transforms
from Helen_transform import Resize, ToPILImage, ToTensor
from model_1 import FaceModel, Stage2FaceModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name_list = ['eyebrows', 'eyes', 'nose', 'mouth']
root_dir = '/home/yinzi/Downloads/datas'
txt_file = 'testing.txt'
# root_dir_2 = "/data1/yinzi/facial_parts"
root_dir_2 = "/home/yinzi/data/facial_parts"
state_file_root = "/home/yinzi/test_model_5/"
state_file_1 = os.path.join(state_file_root,
                            "stage1.pth.tar")
state_file_2 = {x: os.path.join(state_file_root,
                            "{}.pth.tar".format(x))
                for x in model_name_list
                }

# teststage1 = TestStage1(device=device, model_class=FaceModel,
#                         statefile=state_file_1, dataset_class=HelenDataset,
#                         txt_file=txt_file, root_dir=root_dir,
#                         batch_size=4)
#
# teststage1.start_test()

teststage2 = TestStage2(device=device, model_class=Stage2FaceModel,
                        statefile=state_file_2, dataset_class=SinglePart,
                        txt_file=txt_file, root_dir=root_dir_2,
                        batch_size=16)
teststage2.start_test()
