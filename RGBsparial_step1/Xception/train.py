# import os
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from glob import glob
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# import argparse
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from models import model_selection
# # from dataset.transform import xception_default_data_transforms
# from torchvision import transforms
# import torch
# from torch.utils.data import random_split

# # from dataloader import extract_dct, extract_fft


# input_size = 224
# scale = 0.875
# resize_size = int(input_size / scale)

# xception_default_data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize(resize_size),         # 256
#         transforms.RandomHorizontalFlip(),      # train만 augmentation
#         transforms.CenterCrop(input_size),      # 224
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(resize_size),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ]),
#     'test': transforms.Compose([
#         transforms.Resize(resize_size),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ]),
# }


# class EarlyStopping:
#     """Validation loss가 개선되지 않으면 조기 종료."""
#     def __init__(self, patience=5, min_delta=0.0, verbose=False, path='checkpoint_es.pth'):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.verbose = verbose
#         self.counter = 0
#         self.best_loss = np.Inf
#         self.early_stop = False
#         self.checkpoint_path = path

#     def __call__(self, val_loss, model):
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#             # 가장 좋은 모델 저장
#             torch.save(model.state_dict(), self.checkpoint_path)
#             if self.verbose:
#                 print(f"[EarlyStopping] Improved val_loss to {val_loss:.4f}, saving checkpoint.")
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"[EarlyStopping] No improvement in val_loss ({self.counter}/{self.patience}).")
#             if self.counter >= self.patience:
#                 self.early_stop = True

# class FFPPFrameDataset(Dataset):
#     """
#     FaceForensics++ full_images 폴더를 재귀 탐색하여
#     original_sequences/*/{compression}/full_images/** 에 있는 이미지는 label=0 (real),
#     manipulated_sequences/*/{compression}/full_images/** 에 있는 이미지는 label=1 (fake)
#     로 취급합니다.
#     """
#     def __init__(self, root_dir, compression='c23', transform=None, stream_type='rgb'):
#         self.samples = []
#         self.transform = transform
#         self.stream_type = stream_type

#         # real (original_sequences)
#         real_root = os.path.join(root_dir, 'original_sequences')
#         for method in os.listdir(real_root):
#             full_dir = os.path.join(real_root, method, compression, 'faces')
#             if not os.path.isdir(full_dir): continue
#             # full_images 이하 모든 서브폴더 재귀 탐색
#             for subdir, _, files in os.walk(full_dir):
#                 for fname in files:
#                     if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
#                         self.samples.append((os.path.join(subdir, fname), 0))

#         # fake (manipulated_sequences)
#         fake_root = os.path.join(root_dir, 'manipulated_sequences')
#         for method in os.listdir(fake_root):
#             full_dir = os.path.join(fake_root, method, compression, 'faces')
#             if not os.path.isdir(full_dir):
#                 continue
#             for subdir, _, files in os.walk(full_dir):
#                 for fname in files:
#                     if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
#                         self.samples.append((os.path.join(subdir, fname), 1))
        
#         # imgs = sorted(glob(os.path.join(root_dir, '*.jpg')))
#         # # 맨 앞 num_each 개는 label=0, 그다음 num_each 개는 label=1
#         # labels = [0]*50 + [1]*50
#         # # 실제 samples 개수는 min(len(imgs), 2*num_each)
#         # self.samples = list(zip(imgs[:2*50], labels[:len(imgs[:2*50])]))
#         # self.transform = transform
        

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         img = Image.open(path).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img, label

# def train_one_epoch(model, loader, criterion, optimizer, device):
#     model.train()
#     running_loss, running_corrects = 0.0, 0
#     for inputs, labels in tqdm(loader, desc="Train"):
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * inputs.size(0)
#         preds = torch.argmax(outputs, dim=1)
#         running_corrects += (preds == labels).sum().item()
#     return running_loss / len(loader.dataset), running_corrects / len(loader.dataset)

# def validate(model, loader, criterion, device):
#     model.eval()
#     running_loss, running_corrects = 0.0, 0
#     with torch.no_grad():
#         for inputs, labels in tqdm(loader, desc="Val  "):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item() * inputs.size(0)
#             preds = torch.argmax(outputs, dim=1)
#             running_corrects += (preds == labels).sum().item()
#     return running_loss / len(loader.dataset), running_corrects / len(loader.dataset)

# def main(args):
#     device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

#     # 1) 전체 데이터셋 생성 (train용 transform 사용)
#     full_ds = FFPPFrameDataset(
#         root_dir=args.data_root,
#         compression=args.compression,
#         transform=xception_default_data_transforms['train'],
#         stream_type=args.stream_type
#     )

#     # 2) train : val = 90 : 10 비율로 분할
#     total_len = len(full_ds)
#     train_len = int(0.8 * total_len)
#     val_len   = total_len - train_len
#     train_ds, val_ds = random_split(full_ds, [train_len, val_len])

#     # 3) val_ds에는 validation 전용 transform으로 덮어쓰기
#     #    (random_split 으로 나온 Subset 의 dataset 속성에 접근해서 바꿔줍니다)
#     val_ds.dataset.transform = xception_default_data_transforms['val']

#     # 4) DataLoader 생성
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size,
#                               shuffle=True, num_workers=args.workers, pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
#                               shuffle=False, num_workers=args.workers, pin_memory=True)

#     # 2) Model
#     model, image_size, _, _, _ = model_selection('xception', num_out_classes=2)
#     model = model.to(device)
#     if args.freeze_fc:
#         # fc만 trainable
#         model.set_trainable_up_to(False, layername='Conv2d_4a_3x3')

#     # 3) Criterion & Optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
#                                  lr=args.lr, weight_decay=args.weight_decay)
#     # Learning rate 스케줄러 (ReduceLROnPlateau 예시)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=2, verbose=True
#     )

#     # EarlyStopping 인스턴스
#     es_checkpoint = os.path.join(args.out_dir, 'xception_earlystop.pth')
#     early_stopper = EarlyStopping(
#         patience=args.patience,
#         min_delta=args.min_delta,
#         verbose=True,
#         path=es_checkpoint
#     )

#     best_acc = 0.0
#     for epoch in range(1, args.epochs+1):
#         # → 실제 인자를 넘겨야 합니다
#         train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
#         val_loss, val_acc = validate(model, val_loader,  criterion, device)

#         print(f"Epoch {epoch}/{args.epochs} "
#               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
#               f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

#         # 1) LR 스케줄러
#         scheduler.step(val_loss)

#         # 2) EarlyStopping
#         early_stopper(val_loss, model)
#         if early_stopper.early_stop:
#             print("==> Early stopping criterion met. Stopping training.")
#             break

#         # 3) 최고 성능 모델 저장
#         is_best = val_acc > best_acc
#         if is_best:
#             best_acc = val_acc
#             save_path = os.path.join(args.out_dir, "xception_best.pth")
#             torch.save(model.state_dict(), save_path)
#             print(f"→ Saved best model to {save_path}")

#     # 마지막 모델 저장
#     torch.save(model.state_dict(), os.path.join(args.out_dir, "xception_last.pth"))

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--model",      type=str, default="xception",
#                    help="Model: xception / resnet18 / resnet50")
#     p.add_argument("--use-cbam",   action="store_true", help="Use CBAM in Xception")
#     p.add_argument("--pretrained", action="store_true", help="Load ImageNet weights")
#     p.add_argument("--dropout",    type=float, default=0.0, help="Dropout before FC")
#     p.add_argument("--stream-type",type=str, default="rgb",
#                    choices=["rgb","fft","dct"], help="Input stream type")
#     p.add_argument("--data-root",    type=str, required=True,
#                    help="FF++ 루트 경로 (original_sequences, manipulated_sequences 포함)")
#     p.add_argument("--compression",  type=str, default="c23",
#                    help="raw, c23, c40 중 선택")
#     p.add_argument("--epochs",       type=int, default=10)
#     p.add_argument("--batch-size",   type=int, default=32)
#     p.add_argument("--lr",           type=float, default=1e-4)
#     p.add_argument("--weight-decay", type=float, default=1e-5)
#     p.add_argument("--workers",      type=int, default=8)
#     p.add_argument("--cuda",         action="store_true")
#     p.add_argument("--out-dir",      type=str, required=True,
#                    help="모델, 로그 등을 저장할 디렉터리")
#     p.add_argument("--freeze-fc",    action="store_true",
#                    help="처음에는 fc만 학습하고 싶다면 설정")
#     p.add_argument("--patience", type=int, default=5,
#                help="EarlyStopping patience (validation loss 비개선 허용 epoch 수)")
#     p.add_argument("--min-delta", type=float, default=0.0,
#                help="EarlyStopping이 개선으로 간주할 최소 val_loss 감소량")

#     args = p.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)
#     main(args)
