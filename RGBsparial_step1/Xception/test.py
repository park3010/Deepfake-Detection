# # import os
# # import glob
# # import torch
# # from PIL import Image
# # from torchvision import transforms
# # from models import model_selection


# # # def test_models():
# # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # #     print(device)
# # #     batch_size = 4

# # #     for use_cbam in (False, True):
# # #         # 모델 생성
# # #         model = xception(
# # #             num_classes=2,
# # #             pretrained=None,    # 미리 학습된 가중치는 건너뛰기
# # #             use_cbam=use_cbam
# # #         ).to(device)
# # #         model.eval()

# # #         x = torch.randn(batch_size, 3, 224, 224, device=device)

# # #         with torch.no_grad():
# # #             y = model(x)

# # #         # 출력 확인
# # #         print(f"use_cbam={use_cbam} ▶ out.shape = {y.shape}")

# # # if __name__ == "__main__":
# # #     test_models()
# # input_size = 224
# # scale = 0.875
# # resize_size = int(input_size / scale)

# # xception_default_data_transforms = {
# #     'train': transforms.Compose([
# #         transforms.Resize(resize_size),         # 256
# #         transforms.RandomHorizontalFlip(),      # train만 augmentation
# #         transforms.CenterCrop(input_size),      # 224
# #         transforms.ToTensor(),
# #         transforms.Normalize([0.5]*3, [0.5]*3)
# #     ]),
# #     'val': transforms.Compose([
# #         transforms.Resize(resize_size),
# #         transforms.CenterCrop(input_size),
# #         transforms.ToTensor(),
# #         transforms.Normalize([0.5]*3, [0.5]*3)
# #     ]),
# #     'test': transforms.Compose([
# #         transforms.Resize(resize_size),
# #         transforms.CenterCrop(input_size),
# #         transforms.ToTensor(),
# #         transforms.Normalize([0.5]*3, [0.5]*3)
# #     ]),
# # }

# # def main():
# #     # 1) 설정: 이미지 폴더, 모델 옵션
# #     img_dir    = "/home/oem/deepfake/Ourmethod/RGBsparial_step1/Xception/test_data"   # 테스트할 이미지들이 있는 폴더
# #     modelname  = "xception"                   # or "xception" + --use-cbam flag below
# #     use_cbam   = False                        # CBAM 테스트 시 True로 바꿔주세요
# #     pretrained = False                        # 가중치 로딩 확인용
    
# #     # 2) 모델 로드
# #     model, image_size, _, _, _ = model_selection(
# #         modelname,
# #         num_out_classes=2,
# #         pretrained=pretrained,
# #         dropout=0.0,
# #         use_cbam=use_cbam
# #     )
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model = model.to(device).eval()
    
# #     # 3) 전처리: train/val transform 중 val을 사용
# #     transform = xception_default_data_transforms['val']
# #     # (혹은 직접 정의할 경우)
# #     # transform = transforms.Compose([
# #     #     transforms.Resize(int(image_size/0.875)),
# #     #     transforms.CenterCrop(image_size),
# #     #     transforms.ToTensor(),
# #     #     transforms.Normalize([0.5]*3, [0.5]*3)
# #     # ])
    
# #     # 4) 이미지 파일 리스트
# #     img_paths = glob.glob(os.path.join(img_dir, "*.[jJ][pP][gG]")) + \
# #                 glob.glob(os.path.join(img_dir, "*.[pP][nN][gG]"))
# #     print(f"Found {len(img_paths)} images in {img_dir}")
    
# #     # 5) Inference loop
# #     scores = []
# #     with torch.no_grad():
# #         for path in img_paths:
# #             img = Image.open(path).convert("RGB")
# #             inp = transform(img).unsqueeze(0).to(device)   # shape (1,3,H,W)
# #             out = model(inp)                              # shape (1,2)
# #             prob = torch.softmax(out, dim=1)[0,1].item()  # fake 클래스(1번) 확률
# #             scores.append((os.path.basename(path), prob))
    
# #     # 6) 결과 출력 (상위 10개)
# #     scores.sort(key=lambda x: x[1], reverse=True)
# #     print("\nTop 10 images predicted as fake:")
# #     for fn, score in scores[:10]:
# #         print(f"  {fn}: {score:.3f}")
    
# #     print("\nBottom 10 images predicted as fake:")
# #     for fn, score in scores[-10:]:
# #         print(f"  {fn}: {score:.3f}")

# # if __name__ == "__main__":
# #     main()


# import os
# import glob
# import argparse
# from PIL import Image

# import torch
# import torch.nn.functional as F
# from torchvision import transforms

# import sys
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# from xception import xception

# def build_transform(input_size=224, scale=0.875):
#     resize = int(input_size/scale)
#     return transforms.Compose([
#         transforms.Resize(resize),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
#     ])

# def main(args):
#     device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

#     # 2) 모델 로드
#     model = xception(
#         num_classes=2,
#         pretrained='imagenet',
#         use_cbam=args.use_cbam
#     )
#     model = model.to(device).eval()

#     # 3) 이미지 리스트
#     img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.[jp][pn]g")))  # .jpg/.png
#     if len(img_paths)==0:
#         print(">> 이미지 파일이 없습니다:", args.image_dir)
#         return

#     # 4) transform
#     transform = build_transform(input_size=224, scale=0.875)

#     # 5) 순회하며 inference
#     for p in img_paths:
#         img = Image.open(p).convert("RGB")
#         x = transform(img).unsqueeze(0).to(device)   # (1,3,H,W)
#         with torch.no_grad():
#             logits = model(x)
#             probs = F.softmax(logits, dim=1).cpu().squeeze()
#             pred = torch.argmax(probs).item()
#         print(f"{os.path.basename(p):20s}  pred={pred}  conf={probs[pred]:.3f}")

# if __name__=="__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("image_dir", help="테스트할 프레임 이미지 폴더 경로")
#     p.add_argument("--use-cbam",  action="store_true", help="CBAM 적용 모델 사용")
#     p.add_argument("--cuda",      action="store_true", help="GPU 사용")
#     args = p.parse_args()
#     main(args)
