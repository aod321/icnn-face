from torchvision import transforms
from datasets.dataset import HelenDataset, SinglePart, TestStage, TestStage1, TestStage2
from datasets.Helen_transform import ToPILImage, ToTensor
from models.model_1 import FaceModel, Stage2FaceModel
from utils.visualize import save_mask_result, show_mask, apply_mask, random_colors, imshow
import torchvision, torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import numpy as np
import os
import uuid
from skimage import io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name_list = ['eyebrows', 'eyes', 'nose', 'mouth']
root_dir = '/home/yinzi/Downloads/datas'
# root_dir_2 = "/data1/yinzi/facial_parts"
root_dir_2 = "/home/yinzi/data/facial_parts"
state_file_root = "/home/yinzi/test_model_7/"
state_file_1 = os.path.join(state_file_root,
                            "stage1.pth.tar")
state_file_2 = {x: os.path.join(state_file_root,
                                "{}.pth.tar".format(x))
                for x in model_name_list
                }
txt_file = 'testing.txt'


def calc_centroid_single(tensor):
    # Inputs Shape(1, 9 , 64, 64)
    # Return Shape(1, 9 ,2)
    tensor = tensor.float() + 1e-10
    n, l, h, w = tensor.shape
    indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
    indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
    center_y = tensor.sum(3) * indexs_y.view(1, 1, -1)
    center_y = center_y.sum(2, keepdim=True) / tensor.sum([2, 3]).view(n, l, 1)
    center_x = tensor.sum(2) * indexs_x.view(1, 1, -1)
    center_x = center_x.sum(2, keepdim=True) / tensor.sum([2, 3]).view(n, l, 1)
    output = torch.cat([center_y, center_x], 2)
    print(output)
    return output


class TwoStepTest(object):
    def __init__(self):
        self.teststage1 = TestStage1(device=device, model_class=FaceModel,
                                     statefile=state_file_1, dataset_class=HelenDataset,
                                     txt_file=txt_file, root_dir=root_dir,
                                     batch_size=16)
        self.teststage2 = TestStage2(device=device, model_class=Stage2FaceModel,
                                     statefile=state_file_2, dataset_class=SinglePart,
                                     txt_file=txt_file, root_dir=root_dir_2,
                                     batch_size=16
                                     )
        self.unresized_dataset = HelenDataset(txt_file=txt_file,
                                              root_dir=root_dir,
                                              transform=transforms.Compose([
                                                  ToPILImage(),
                                                  ToTensor()
                                              ]))
        self.parts_name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        self.model_name_list = ['eyebrows', 'eyes', 'nose', 'mouth']
        self.centroids = None
        self.final_mask = None
        self.crop_rect_list = None

    def twostep_test(self):
        for i_batch, sample_batched in enumerate(self.teststage1.dataloader):
            img = sample_batched['image'].to(device)
            labels = sample_batched['labels'].to(device)
            idx = sample_batched['index']
            self.teststage1.get_predict(self.teststage1.model, img)
            calculate_centroids(idx, self.teststage1.predict)
            # Centroids 10 x 9 x 2

    def single_image_test(self, img):
        # Input Image shape(3,h,w)
        # orig_image = TF.to_tensor(TF.to_pil_image(img))
        n, h, w = img.shape
        image = TF.to_tensor(TF.resize(TF.to_pil_image(img), (64, 64), Image.ANTIALIAS))
        image = image.to(self.teststage1.device).unsqueeze(0)  # Shape(1, 3, 64, 64)
        # Stage1 Predict shape(1, 9, 64, 64)
        stage1_pred = torch.softmax(self.teststage1.model(image), 1)
        refer = stage1_pred.argmax(dim=1, keepdim=False)  # Shape(1, 64, 64)
        np_ref = refer[0].float().detach().cpu().numpy()

        # Convert to one-hot
        for i in range(stage1_pred.shape[1]):
            stage1_pred[:, i] = (refer == i).float()  # Shape(1, 9, 64, 64)
        temp_resize = []
        for i in range(stage1_pred.shape[1]):
            temp_image = TF.to_tensor(TF.resize(TF.to_pil_image(stage1_pred[0, i].detach().cpu()), (h, w))) # List 9 x Shape(H, W)
            temp_resize.append(temp_image)
        temp_resize = torch.stack(temp_resize).transpose(1, 0)
        centroids = calc_centroid_single(temp_resize)  # Shape(1, 9, 2)
        orig_img = img.unsqueeze(0)
        # Extracted parts
        # Input centroids Shape(1, 9, 2) img Shape(3, H, W)
        # Output parts image Shape(1, 5, 3, 64, 64) and (1, 1, 3, 80, 80)
        extracted_parts, crop_list = self.extract_batch_image(centroids, orig_img)
        self.crop_rect_list = crop_list
        # Stage2_baches Non-mouth Shape(1, 5, 3, 64, 64)  Mouth Shape(1, 1, 3, 80, 80)
        stage2_bacthes = self.prepare_bacthes(extracted_parts)
        """
          Eyes Shape(1, 2, 3, 64, 64)
          Eyebrows Shape(1, 2, 3, 64, 64)
          Nose Shape(1, 1, 3, 64, 64)
          Mouth Shape(1, 1, 3, 80, 80)
        """
        stage2_preds = self.get_stage2_predicts(stage2_bacthes)  # ex: pred['eyes'] Shape(2, 1, 2, 64, 64)
        # Stage2_results not-mouth: Shape(N, 6, 64, 64) mouth: Shape(N, 4, 80, 80)
        stage2_results = self.get_stage2_results(stage2_preds)
        final_mask = self.combine_mask(orig_img, stage2_results, centroids)
        self.show_results(orig_img, final_mask)

    def get_stage2_predicts(self, stage2_bacthes):
        # Input
        # Eyes Shape(N, 2, 3, 64, 64)
        # Eyebrows Shape(N, 2, 3, 64, 64)
        # Nose Shape(N, 1, 3, 64, 64)
        # Mouth Shape(N, 1, 3, 80, 80)

        # Output
        stage2_preds = {r: []
                        for r in self.model_name_list}
        for x in self.model_name_list:
            for i in range(stage2_bacthes[x].shape[1]):
                stage2_preds[x].append(self.teststage2.get_predict_onehot(self.teststage2.model[x],
                                                                          stage2_bacthes[x][:, i]))
            # Stage2_preds ex: eyes_pred List 2 x Shape(N, 2, 64, 64)
            stage2_preds[x] = torch.stack(stage2_preds[x])  # eyes_pred Shape(2, N, 2, 64, 64)
        return stage2_preds

    def get_f1_scores(self, stage2_preds, labels):
        # stage2_preds nose: Shape(1, N, 2, 64, 64) mouth: Shape(1, N, 4, 80, 80) double_parts Shape(2, N, 2, 64, 64)
        # labels Shape(N, 2, 64, 64) or Shape(N, 4, 80, 80)
        for x in self.model_name_list:
            for i in range(stage2_preds[x].shape[0]):
                # F1 {x: [f1_1, ... , f1_N]}
                f1 = self.teststage2.calc_f1(x=x, predict=stage2_preds[i], labels=labels)
        f1_result = self.teststage2.output_f1_score(f1)
        return f1_result

    def get_stage2_results(self, stage2_preds):
        # eyes_pred Shape(2, N, 2, 64, 64)
        # Nose Shape(1, N , 2, 64, 64)
        # Mouth Shape(1, N , 4, 80, 80)

        # Output Non-mouth Shape(N, 5, 64, 64) Mouth Shape(N, 3 ,80, 80)
        n = stage2_preds['eyebrows'].shape[1]
        stage2_results = {r: []
                          for r in self.parts_name_list}
        results = {'not_mouth': [], 'mouth': []}
        stage2_results['eyebrow1'] = stage2_preds['eyebrows'][0]
        stage2_results['eyebrow2'] = torch.stack([TF.to_tensor(
            TF.hflip(
                TF.to_pil_image(
                    stage2_preds['eyebrows'][1, i].detach().cpu()
                ))
        )
            for i in range(n)
        ])
        stage2_results['eye1'] = stage2_preds['eyes'][0]
        stage2_results['eye2'] = torch.stack([TF.to_tensor(
            TF.hflip(
                TF.to_pil_image(
                    stage2_preds['eyes'][1, i].detach().cpu()
                ))
        )
            for i in range(n)
        ])
        stage2_results['nose'] = stage2_preds['nose'][0]
        stage2_results['mouth'] = stage2_preds['mouth'][0]
        # stage2_results Shape(N, 2, 64, 64) or Shape(N, 4, 80, 80)
        # fg results not_mouth Shape(N, 5, 64, 64)
        results['not_mouth'] = [stage2_results[r][:, 1].to(device)
                                for r in ['eyebrow1', 'eyebrow2',
                                          'eye1', 'eye2', 'nose']]
        # final results not-mouth: Shape(N, 5, 64, 64) mouth: Shape(N, 3, 80, 80        results['mouth'] = stage2_results['mouth'][1:4]  # mouth: Shape(N, 3, 80, 80))
        results['not_mouth'] = torch.stack(results['not_mouth'], dim=0).transpose(1, 0)
        results['mouth'] = stage2_results['mouth'][:, 1:4]  # mouth: Shape(N, 3, 80, 80)
        return results

    def combine_mask(self, orig_img, pred_results, centroids):
        # Combine all orig-size mask
        # input image Shape(N, 3, H, W)
        # centroids Shape(N, 9, 2)  pred_results  not-mouth: Shape(N, 5, 64, 64) mouth: Shape(N, 3, 80, 80)
        # Output Shape(N, 8, H, W)

        # make sure orig_img is in Tensor format   (C,H,W)
        box_size = 1024
        n, c, h, w = orig_img.shape
        offset_h, offset_w = int((box_size - h) // 2), int((box_size - w) // 2)
        remap_mask = torch.zeros(n, 8, box_size, box_size)
        _, c_1, h_1, w_1 = pred_results['not_mouth'].shape
        _, c_2, h_2, w_2 = pred_results['mouth'].shape
        # not_mouth_centroids = calc_centroid_single(pred_results['not_mouth'])  # Shape(N, 5, 2)
        # mouth_centroids = calc_centroid_single(pred_results['mouth'])   # Shape(N, 3, 2)
        # pred_centroids = torch.cat([not_mouth_centroids, mouth_centroids], dim=1)  # Shape(N, 8, 2)
        for i in range(n):
            # Not mouth
            for j in range(c_1):
                h_start, w_start = self.crop_rect_list[j]
                h_start = int(h_start)
                w_start = int(w_start)
                np_n_mouth = pred_results['not_mouth'][i, j].detach().cpu().numpy()
                pred_n_mouth = Image.fromarray(np_n_mouth)
                left, upper, right, lower = pred_n_mouth.getbbox()
                old_bbox = (left, upper, right, lower)
                pred_n_mouth = pred_n_mouth.crop(old_bbox)
                new_bbox = (w_start + left, h_start + upper, w_start + right, h_start + lower)
                img1 = Image.fromarray(remap_mask[i, j].detach().cpu().numpy())
                img1.paste(pred_n_mouth, box=new_bbox)
                remap_mask[i, j] = torch.from_numpy(np.array(img1))
            # Mouth
            for j in range(c_2):
                k = j + c_1
                h_start, w_start = self.crop_rect_list[5]
                h_start = int(h_start)
                w_start = int(w_start)
                np_mouth = pred_results['mouth'][i, j].detach().cpu().numpy()
                pred_mouth = Image.fromarray(np_mouth)
                left, upper, right, lower = pred_mouth.getbbox()
                old_bbox = (left, upper, right, lower)
                pred_mouth = pred_mouth.crop(old_bbox)
                new_bbox = (w_start + left, h_start + upper, w_start + right, h_start + lower)
                img1 = Image.fromarray(remap_mask[i, k].detach().cpu().numpy())
                img1.paste(pred_mouth, box=new_bbox)
                remap_mask[i, k] = torch.from_numpy(np.array(img1))

        final_mask = remap_mask[:, :, offset_h:offset_h + h, offset_w:offset_w + w]
        self.final_mask = final_mask
        return final_mask

    def get_masked_orig_img(self, img, final_mask):
        # img Shape(N, 3, H, W) final_mask Shape(N, 8, H ,W)
        # Out masked_img Shape(N, 3, H, W)
        n, c, h, w = final_mask.shape
        colors = random_colors(c)
        img_list = []
        for i in range(n):
            image_masked = np.array(TF.to_pil_image(img[i].detach().cpu()))
            for k in range(c):
                color = colors[k]
                image_masked = apply_mask(image=image_masked,
                                          mask=final_mask[i][k], color=color, alpha=0.5)
            image_masked = np.array(image_masked.clip(0, 255), dtype=np.uint8)
            img_list.append(TF.to_tensor(TF.to_pil_image(image_masked)))
        out_img = torch.stack(img_list)
        return out_img

    def show_results(self, img, final_mask, title=None):
        # img Shape(N, 3, H, W) final_mask Shape(N, 8, H ,W)
        masked_img = self.get_masked_orig_img(img, final_mask)
        out = torchvision.utils.make_grid(masked_img)
        imshow(out, title)

    def save_results(self, img, final_mask, title=None):
        # Save original image with mask
        masked_img = self.get_masked_orig_img(img, final_mask)
        out = torchvision.utils.make_grid(masked_img)
        out = out.detach().cpu().numpy().transpose((1, 2, 0))
        out = np.clip(out, 0, 1)
        save_dir = os.path.join("two_step_res")
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        # Random name
        uuid_str = str(uuid.uuid4().hex)
        plt.imshow(masked_img)
        if title is not None:
            plt.title(title)
        plt.savefig(os.path.join(save_dir, '{}.png'.format(uuid_str)))

    def extract_batch_image(self, cens, img):
        # Centroids Shape(N, 9, 2)   Input image Shape(N, 3, H, W)
        # get output parts image Shape(N, 5, 3, 64, 64) Shape(N, 1, 3, 80, 80)
        box_size = 1024
        n, c, h, w = img.shape
        offset_h, offset_w = (box_size - h) // 2, (box_size - w) // 2
        image = torch.zeros(n, 3, box_size, box_size)
        image[:, :, offset_h:offset_h + h, offset_w:offset_w + w] = img
        parts = {'mouth': torch.zeros((n, 1, 3, 80, 80)),
                 'not_mouth': torch.zeros((n, 5, 3, 64, 64))}
        h_1, w_1 = 64, 64
        h_2, w_2 = 80, 80
        mouth_cen = torch.floor(cens[:, 6:9].mean(dim=1, keepdim=True))
        centroids = torch.cat([cens[:, 0:6], mouth_cen], dim=1)  # Shape(N, 7 ,2)
        crop_rect_list = []
        print("offset")
        print(offset_h, offset_w)
        for i in range(n):
            for j in range(5):
                k = j + 1
                index_h = centroids[i, k, 0] + offset_h
                index_w = centroids[i, k, 1] + offset_w
                h_start = torch.floor(index_h - h_1 // 2).int().tolist()
                h_end = torch.floor(index_h + h_1 // 2).int().tolist()
                w_start = torch.floor(index_w - w_1 // 2).int().tolist()
                w_end = torch.floor(index_w + w_1 // 2).int().tolist()
                crop_rect_list.append((h_start, w_start))
                print(j, h_start, h_end, w_start, w_end)
                parts['not_mouth'][i, j, :] = image[i, :, h_start:h_end, w_start:w_end]
                print(parts['not_mouth'][i, j, :].shape)
            index_h = centroids[i, 6, 0] + offset_h
            index_w = centroids[i, 6, 1] + offset_w
            h_start = torch.floor(index_h - h_2 // 2).int().tolist()
            h_end = torch.floor(index_h + h_2 // 2).int().tolist()
            w_start = torch.floor(index_w - w_2 // 2).int().tolist()
            w_end = torch.floor(index_w + w_2 // 2).int().tolist()
            print(h_start, h_end, w_start, w_end)
            parts['mouth'][i, 0, :] = image[i, :, h_start:h_end, w_start:w_end]
            crop_rect_list.append((h_start, w_start))
        return parts, crop_rect_list

    def prepare_bacthes(self, parts):
        # Input parts not_Mouth Shape(N, 5, 3, 64, 64) and Mouth Shape(N, 1, 3, 80, 80)
        # Eyes Shape(N, 2, 3, 64, 64)
        # Eyebrows Shape(N, 2, 3, 64, 64)
        # Nose Shape(N, 1, 3, 64, 64)
        # Mouth Shape(N, 1, 3, 80, 80)
        n = parts['not_mouth'].shape[0]
        eyebrow1 = parts['not_mouth'][:, 0:1]
        eyebrow2 = parts['not_mouth'][:, 1:2]  # Shape(N, 1, 3, 64, 64)
        eyebrow2 = torch.stack([TF.to_tensor(TF.hflip(TF.to_pil_image(eyebrow2[i, 0])))
                            for i in range(n)]).unsqueeze(1)
        eyebrows = torch.cat([eyebrow1, eyebrow2], dim=1)  # Shape(N, 2, 3, 64, 64)
        eye1 = parts['not_mouth'][:, 2:3]
        eye2 = parts['not_mouth'][:, 3:4]  # Shape(N, 1, 3, 64, 64)
        eye2 = torch.stack([TF.to_tensor(TF.hflip(TF.to_pil_image(eye2[i, 0])))
                                for i in range(n)]).unsqueeze(1)
        eyes = torch.cat([eye1, eye2], dim=1)
        bacthes = {'eyes': eyes,
                   'eyebrows': eyebrows,
                   'nose': parts['not_mouth'][:, 4:5],
                   'mouth': parts['mouth']
                   }
        return bacthes

    def get_centroids(self, index, y):
        inputs = y.float() + 1e-20
        refer = y.argmax(dim=1, keepdim=False)
        # Make predicts only have 0 and 1
        for i in range(y.shape[1]):
            inputs[:, i] = (refer == i)
        inputs = inputs.float()
        inputs = inputs.detach().cpu().numpy()
        inputs = np.array(inputs, np.float32)
        centroids_list = []
        for i, idx in enumerate(index):
            n, l, h, w = self.unresized_dataset[idx]['labels']
            temp = [cv2.resize(inputs[i][j], (w, h))
                    for j in range(y.shape[1])]
            temp = np.array(temp, np.float32)  # (9, h, w)
            tensor = torch.from_numpy(temp)  # (9, h, w)
            l1, h1, w1 = tensor.shape
            tensor = tensor.view(1, l1, h1, w1)  # (1, 9, h, w)
            centroids_single = calc_centroid_single(tensor)  # (1, 9, 2)
            centroids_list.append(centroids_single)

        self.centroids = torch.cat(centroids_list, 0)  # (10, 9, 2)
        return self.centroids

image = np.array(io.imread('test.jpg'))
h, w, n = image.shape
image = TF.to_pil_image(image)
if w > 600 or h > 600:
    image.thumbnail((256, 256), Image.ANTIALIAS)
image.show()
print(image.size)
image = TF.to_tensor(image)
two_step = TwoStepTest()
two_step.single_image_test(image)
