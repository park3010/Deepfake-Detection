import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1) 파일 경로 설정
LABEL_CSV      = Path('/home/oem/deepfake/hdd/labels.csv')
SUBMISSION_CSV = Path('/home/oem/deepfake/Ourmethod/Frequency_step2/test_csv/convnext_fft_frame5_fixed.csv')

# 2) 데이터 읽기 (ID, label)
df_gt  = pd.read_csv(LABEL_CSV,      dtype={'ID':str,'label':int})   # ['ID','label']
df_sub = pd.read_csv(SUBMISSION_CSV, dtype={'ID':str,'label':int})   # ['ID','label']

# 3) 컬럼명 통일
df_gt  = df_gt .rename(columns={'label':'label_true'})
df_sub = df_sub.rename(columns={'label':'label_pred'})

df_sub['ID'] = df_sub['ID'].str.replace(r'\.mp4$', '', regex=True)

# 4) ID 공백·소문자 정리 (선택)
df_gt ['ID'] = df_gt ['ID'].str.strip().str.lower()
df_sub['ID'] = df_sub['ID'].str.strip().str.lower()

# 5) merge 전에 컬럼 체크
print("GT columns :", df_gt.columns.tolist())
print("Sub columns:", df_sub.columns.tolist())

# 6) ID 기준으로 merge
df_merged = (
    df_sub
    .merge(df_gt, on='ID', how='left')
)

# 7) 매칭 실패한 ID 확인
missing = df_merged['label_true'].isna().sum()
if missing:
    print(f"⚠️ WARNING: {missing}개의 ID에 대한 정답을 찾을 수 없습니다.")
    print(df_merged.loc[df_merged['label_true'].isna(), 'ID'].tolist()[:10])

# 8) metric 계산
y_true = df_merged['label_true'].astype(int)
y_pred = df_merged['label_pred'].astype(int)

# 9) ID, kaggle(실제), submission(예측) 동시 저장
out_df = pd.DataFrame({
    'ID':            df_merged['ID'],
    'kaggle_label':  y_true,
    'my_prediction': y_pred,
})
out_csv = Path('/home/oem/deepfake/Ourmethod/real_pred.csv')
out_df.to_csv(out_csv, index=False)
print(f"✅ Saved combined CSV to {out_csv}")

# 10) 성능 출력
print("=== Test-set Performance ===")
print(f"Accuracy : {accuracy_score(y_true, y_pred)*100:6.2f}%")
print(f"Precision: {precision_score( y_true, y_pred, average='macro')*100:6.2f}%")
print(f"Recall   : {recall_score(    y_true, y_pred, average='macro')*100:6.2f}%")
print(f"F1 score : {f1_score(      y_true, y_pred, average='macro')*100:6.2f}%")
