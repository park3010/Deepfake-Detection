import os, argparse, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from PIL import Image, UnidentifiedImageError

#from Xception.xception import xception
#from maxvit.maxvit import MaxViT
from hornet.hornet import hornet_tiny_gf, hornet_base_gf
#from coatnet.coatnet import coatnet_0

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

# ------------------------- FF++ 경로 매핑 ------------------------
DATASETS = {
    'original':                     'original_sequences/youtube',
    'DeepFakeDetection_original':   'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face':      'manipulated_sequences/Face2Face',
    'FaceShifter':    'manipulated_sequences/FaceShifter',
    'FaceSwap':       'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures',
}

EXCLUDE_DATASETS = {'Deepfakes', 'DeepFakeDetection'}

# ====================== Early-Stopping 클래스 =====================
class EarlyStopping:
    """`val_loss`(작을수록 좋음)이 `patience` epoch 동안 개선되지 않으면 조기 종료"""
    def __init__(self, patience=5, min_delta=0.0,
                 verbose=False, path='checkpoint_es.pth'):
        self.patience       = patience
        self.min_delta      = min_delta
        self.verbose        = verbose
        self.counter        = 0
        self.best_loss      = np.inf
        self.early_stop     = False
        self.checkpoint_path= path

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                print(f"[EarlyStopping] val_loss improved → {val_loss:.4f} "
                      f"(ckpt saved)")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] no improve ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                

class FFPP_RGB(torch.utils.data.Dataset):
    def __init__(self, root_dir, compression='c23', transform=None, split_half=True):
        self.t = transform
        self.samples = []
        roots = [os.path.join(root_dir, 'original_sequences'),
                 os.path.join(root_dir, 'manipulated_sequences')]
        
        for label, base in enumerate(roots):
            for method in os.listdir(base):
                if base.endswith('manipulated_sequences') and method in EXCLUDE_DATASETS:
                     continue
                d = os.path.join(base, method, compression, 'mtcnn')
                if not os.path.isdir(d): continue
                for sub,_,fs in os.walk(d):
                    for f in fs:
                        if f.lower().endswith(('png','jpg','jpeg')):
                            self.samples.append((os.path.join(sub,f), label))

        
    def __len__(s): return len(s.samples)
    def __getitem__(s, i):
        p,l = s.samples[i]
        try:
            img = Image.open(p).convert('RGB')
        except (UnidentifiedImageError, OSError) as e:
            print(f"Warning: cannot open image {p}, skipping ({e})")
            return None  # DataLoader 쪽에서 걸러내도록

        if s.t: img = s.t(img)
        return img, l


def load_model(name:str, use_cbam:bool):
    if name == 'xception':
        return xception(num_classes=2, use_cbam=use_cbam)
    if name == 'maxvit':
        return MaxViT(num_classes=2, use_cbam=use_cbam)
    if name == 'hornet':
        return hornet_base_gf(num_classes=2, use_cbam=use_cbam)
    if name == 'coatnet':
        return coatnet_0(num_classes=2)
    raise ValueError(f'Unsupported model {name}')


def compute_metrics(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(y.numpy().tolist())
    return {
        'f1':      f1_score(trues, preds, average='macro'),
        'prec':   precision_score(trues, preds, average='macro', zero_division=0),
        'recall':  recall_score(trues, preds, average='macro', zero_division=0),
        'acc':    accuracy_score(trues, preds)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['xception','maxvit','hornet','coatnet'])
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch',  type=int, default=32)
    ap.add_argument('--lr',     type=float, default=3e-4)
    ap.add_argument('--use-cbam', action='store_true')
    ap.add_argument('--mode', choices=['train','val'], default='train')
    ap.add_argument('--ckpt', type=str, help='--mode val 일 때 불러올 체크포인트')
    ap.add_argument('--patience',  type=int, default=3, help='Early-Stopping patience')
    ap.add_argument('--min-delta', type=float, default=0.0, help='Early-Stopping min_delta')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tf  = transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.5]*3,[0.5]*3)])

    ds = FFPP_RGB(args.data_dir, transform=tf)
    print(f"Total frames : {len(ds):,}")
    tr_len = int(.8*len(ds)); va_len = len(ds)-tr_len
    tr,va = random_split(ds,[tr_len,va_len], generator=torch.Generator().manual_seed(42))
    tr_ld = DataLoader(tr,args.batch,True ,num_workers=2,pin_memory=False, collate_fn=lambda batch: default_collate([b for b in batch if b is not None]))
    va_ld = DataLoader(va,args.batch,False,num_workers=2,pin_memory=False, collate_fn=lambda batch: default_collate([b for b in batch if b is not None]))

    model = load_model(args.model, args.use_cbam).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model {args.model}{'+CBAM' if args.use_cbam else ''} "
          f"→ {n_params:.1f} M params")
    
    # ------------------ 파일/이름 ----------------
    os.makedirs('checkpoints', exist_ok=True)
    tag = f"{args.model}{'_cbam' if args.use_cbam else ''}"
    best_ckpt = f"checkpoints/{tag}_best.pth"
    es_ckpt   = f"checkpoints/{tag}_earlystop.pth"

    # ------------------ Early-Stopping -----------
    early_stopper = EarlyStopping(patience=args.patience,
                                  min_delta=args.min_delta,
                                  verbose=True,
                                  path=es_ckpt)
    
    
    # train
    if args.mode == 'train':
        crit = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        
        best_f1 = 0.0
        for epoch in range(1, args.epochs+1):
            model.train(); running_loss = 0.
            pbar = tqdm(tr_ld, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
            for x, y in tqdm(tr_ld, desc="Train 배치 처리", leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = crit(model(x), y)
                loss.backward(); optimizer.step()
                running_loss += loss.item()
                
            train_metric = compute_metrics(model, tr_ld, device)
            print(f"\n[Epoch {epoch}] "
                  f"Train loss {running_loss/len(tr_ld):.4f} | "
                  f"Acc {train_metric['acc']:.3f}  F1 {train_metric['f1']:.3f}")

            val_metric = compute_metrics(model, va_ld, device)
            print(f"[Epoch {epoch}] "
                  f"Val   Acc {val_metric['acc']:.3f}  F1 {val_metric['f1']:.3f} "
                  f"Prec {val_metric['prec']:.3f}  Recall {val_metric['recall']:.3f}")

            # -------- Early-Stopping ----------
            val_loss = 1 - val_metric['f1']
            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                print(f">>> Early stopped! ckpt = {early_stopper.checkpoint_path}")
                break

            # -------- Best F1 저장 ------------
            if val_metric['f1'] > best_f1:
                best_f1 = val_metric['f1']
                torch.save(model.state_dict(), best_ckpt)
                print(f"  ↑ Best F1 {best_f1:.4f} (saved to {best_ckpt})")
                
        print(f"Best ckpt: {best_ckpt}, F1: {best_f1:.4f}")
        
        
    else:
        assert args.ckpt, "--mode val 인 경우 --ckpt 필요"
        model.load_state_dict(torch.load(args.ckpt, map_location=device))

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(va_ld, desc="Validate"):
                logits = model(x.to(device))
                probs = torch.softmax(logits, 1).detach().cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y.numpy())
                
        probs  = np.vstack(all_probs)
        labels = np.concatenate(all_labels)
        os.makedirs('result', exist_ok=True)
        np.save(f'result/{tag}_exclude_large_rgb_probs.npy', probs)
        np.save(f'result/{tag}_exclude_large_y_true.npy',   labels)

        preds = probs.argmax(1)
        print("=== Validation Metrics ===")
        print("Accuracy :", accuracy_score(labels, preds))
        print("F1 score :", f1_score(labels, preds, average='macro'))
        print("Precision:", precision_score(labels, preds, average='macro', zero_division=0))
        print("Recall   :", recall_score(labels, preds, average='macro', zero_division=0))


if __name__ == '__main__':
    main()
