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


def calc_centroids(tensor):
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
    # print(output)
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
        self.crop_rect = None
        self.orig_img = None

    def start_all_test(self):
        for i_batch, sample_batched in enumerate(self.teststage1.dataloader):
            img = sample_batched['image'].to(device)
            index = sample_batched['index']
            stage1_pred = self.teststage1.get_predict_onehot(self.teststage1.model, img)
            self.get_stage1_centroids(index=index, stage1_pred=stage1_pred)
            self.orig_img = {i:
                                 self.unresized_dataset[idx]['image']
                             for i, idx in enumerate(index)
                             }
            extracted_parts, crop_list = self.extract_batch_image(self.centroids, self.orig_img)
            stage2_bacthes = self.prepare_bacthes(extracted_parts)
            stage2_preds = self.get_stage2_predicts(stage2_bacthes)
            stage2_results = self.get_stage2_results(stage2_preds)
        f1 = self.get_f1_scores(stage2_preds)
        print(f1)
            # final_mask = self.combine_mask(orig_img=self.orig_img,
            #                                pred_results=stage2_results,
            #                                centroids=self.centroids)
            # self.show_results(final_mask)

    def single_image_test(self, img, colors=None):
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
            temp_image = TF.to_tensor(
                TF.resize(TF.to_pil_image(stage1_pred[0, i].detach().cpu()), (h, w)))  # List 9 x Shape(H, W)
            temp_resize.append(temp_image)
        temp_resize = torch.stack(temp_resize).transpose(1, 0)
        centroids = calc_centroids(temp_resize)  # Shape(1, 9, 2)
        orig_img = img.unsqueeze(0)
        # Extracted parts
        # Input centroids Shape(1, 9, 2) img Shape(3, H, W)
        # Output parts image Shape(1, 5, 3, 64, 64) and (1, 1, 3, 80, 80)
        extracted_parts, crop_list = self.extract_batch_image(centroids, orig_img)

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
        # self.show_results(orig_img, final_mask)
        masked_image = self.get_masked_orig_img(orig_img,final_mask, colors)
        return masked_image

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
                f1 = self.teststage2.calc_f1(x=x, predict=stage2_preds[x][i], labels=labels)
        f1_result = self.teststage2.output_f1_score(f1)
        return f1_result

    def get_stage2_results(self, stage2_preds):
        # eyes_pred Shape(2, N, 2, 64, 64)
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
        # final results not-mouth: Shape(N, 5, 64, 64)
        # mouth: Shape(N, 3, 80, 80
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
        n = len(orig_img)
        remap_mask = torch.zeros(n, 8, box_size, box_size)
        _, c_1, h_1, w_1 = pred_results['not_mouth'].shape
        _, c_2, h_2, w_2 = pred_results['mouth'].shape
        c, h, w = orig_img[0].shape
        offset_h, offset_w = int((box_size - h) // 2), int((box_size - w) // 2)
        for i in range(n):
            c, h, w = orig_img[i].shape
            offset_h, offset_w = int((box_size - h) // 2), int((box_size - w) // 2)
            # Not mouth
            for j in range(c_1):
                h_start, w_start = self.crop_rect[i][j]
                h_start = int(h_start)
                w_start = int(w_start)
                np_n_mouth = pred_results['not_mouth'].detach().cpu().numpy()
                pred_n_mouth = Image.fromarray(np_n_mouth[i, j])
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
                h_start, w_start = self.crop_rect[i][5]
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

    def get_masked_orig_img(self, img, final_mask, colors=None):
        # img Shape(N, 3, H, W) final_mask Shape(N, 8, H ,W)
        # Out masked_img Shape(N, 3, H, W)
        n, c, h, w = final_mask.shape
        if colors is None:
            colors = random_colors(c)
        img_list = []
        for i in range(n):
            image_masked = np.array(TF.to_pil_image(img[i].detach().cpu()))
            for k in range(c):
                color = colors[k]
                image_masked = apply_mask(image=image_masked,
                                          mask=final_mask[i][k], color=color, alpha=0.4)
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

    def extract_labels(self,):

    def extract_batch_image(self, cens, img):
        # Centroids Shape(N, 9, 2)   Input image Shape(N, 3, H, W)
        # get output parts image Shape(N, 5, 3, 64, 64) Shape(N, 1, 3, 80, 80)
        box_size = 1024
        n = len(img)
        parts = {'mouth': torch.zeros((n, 1, 3, 80, 80)),
                 'not_mouth': torch.zeros((n, 5, 3, 64, 64))}
        h_1, w_1 = 64, 64
        h_2, w_2 = 80, 80
        mouth_cen = torch.floor(cens[:, 6:9].mean(dim=1, keepdim=True))
        centroids = torch.cat([cens[:, 0:6], mouth_cen], dim=1)  # Shape(N, 7 ,2)
        temp_list = []
        for i in range(n):
            crop_rect_list = []
            c, h, w = img[i].shape
            offset_h, offset_w = (box_size - h) // 2, (box_size - w) // 2
            image = torch.zeros(3, box_size, box_size)
            image[:, offset_h:offset_h + h, offset_w:offset_w + w] = img[i]
            # test_show = TF.to_pil_image(image.detach().cpu())
            # test_show.show()
            for j in range(5):
                k = j + 1
                index_h = centroids[i, k, 0] + offset_h
                index_w = centroids[i, k, 1] + offset_w
                h_start = torch.floor(index_h - h_1 // 2).int().tolist()
                h_end = torch.floor(index_h + h_1 // 2).int().tolist()
                w_start = torch.floor(index_w - w_1 // 2).int().tolist()
                w_end = torch.floor(index_w + w_1 // 2).int().tolist()
                crop_rect_list.append((h_start, w_start))
                parts['not_mouth'][i, j, :] = image[:, h_start:h_end, w_start:w_end]
                # print(parts['not_mouth'][i, j, :].shape)
                # cv2.imshow('ex_%d' % j, np.array(TF.to_pil_image(parts['not_mouth'][i, j, :].detach().cpu())))
                # cv2.waitKey()
            index_h = centroids[i, 6, 0] + offset_h
            index_w = centroids[i, 6, 1] + offset_w
            h_start = torch.floor(index_h - h_2 // 2).int().tolist()
            h_end = torch.floor(index_h + h_2 // 2).int().tolist()
            w_start = torch.floor(index_w - w_2 // 2).int().tolist()
            w_end = torch.floor(index_w + w_2 // 2).int().tolist()
            # print(h_start, h_end, w_start, w_end)
            parts['mouth'][i, 0, :] = image[:, h_start:h_end, w_start:w_end]
            # cv2.imshow('mouth', np.array(TF.to_pil_image(parts['mouth'][i, 0, :].detach().cpu())))
            # cv2.waitKey()
            crop_rect_list.append((h_start, w_start))  # Size: 6 x 2
            temp_list.append(np.array(crop_rect_list))
        crop_rect = torch.from_numpy(np.array(temp_list))  # Shape(N, 6, 2)
        self.crop_rect = crop_rect
        return parts, crop_rect

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

    def get_stage1_centroids(self, index, stage1_pred):
        centroids_list = []
        for i, idx in enumerate(index):
            orig_img = self.unresized_dataset[idx]['image']
            n, h, w = orig_img.shape
            # show_test = TF.to_pil_image(orig_img)
            # np_show = nparray(show_test)
            # cv2.imshow('show_orig', np_show)
            # cv2.waitKey()
            temp_image = [TF.to_tensor(
                TF.resize(TF.to_pil_image(
                    stage1_pred[i, j].detach().cpu()
                ), (h, w)))
                for j in range(stage1_pred.shape[1])]
            temp_image = torch.stack(temp_image).transpose(1, 0)  # Shape(1, 9, H, W)
            temp_centroids = calc_centroids(temp_image)  # Shape(1, 9, 2)
            # for p in range(9):
            #     y = temp_centroids[0, p, 0]
            #     x = temp_centroids[0, p, 1]
            #     draw = ImageDraw.Draw(show_test)
            #     draw.point((x, y), fill=(255, 0, 0))
            # np_show = np.array(show_test)
            # cv2.imshow('show_after', np_show)
            # cv2.waitKey()
            centroids_list.append(temp_centroids)
        centroids = torch.cat(centroids_list, 0)  # Shape(N, 9, 2)
        self.centroids = centroids
        return centroids


# image = np.array(io.imread('test.jpg'))

def real_time():
    two_step = TwoStepTest()
    capture = cv2.VideoCapture(0)
    colors = random_colors(8)
    while True:
        ref, frame = capture.read()
        after = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        n , h, w = after.shape
        after = TF.to_pil_image(after)
        if w > 600 or h > 600:
            after.thumbnail((400, 400), Image.ANTIALIAS)
        after = TF.to_tensor(after)
        masked_image = two_step.single_image_test(after, colors)
        masked_image = np.array(TF.to_pil_image(masked_image[0]))
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', masked_image)
        c = cv2.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break


two_step = TwoStepTest()
two_step.start_all_test()

# h, w, n = image.shape
# image = TF.to_pil_image(image)
# if w > 600 or h > 600:
#     image.thumbnail((256, 256), Image.ANTIALIAS)
# image.show()
# print(image.size)
# image = TF.to_tensor(image)
# two_step.single_image_test(image)
