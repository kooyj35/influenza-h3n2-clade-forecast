import pandas as pd
import numpy as np
import re
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# ================================================================
# 1. 설정 및 항원 결정 부위(Epitope) 정의
# ================================================================

# H3N2 HA1 Epitope Sites (Wiley et al. 1981 / Koel et al. 2013 기준)
# 항원성 변화에 직접적인 영향을 주는 부위들입니다.
ALL_EPITOPE_SITES = {
    122, 124, 126, 130, 131, 132, 133, 135, 137, 138, 140, 142, 143, 144, 145, 146, 150, 152, 168, # A
    128, 129, 155, 156, 157, 158, 159, 160, 163, 164, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198, # B
    44, 45, 46, 47, 48, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312, # C
    96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 220, 222, 223, 224, 225, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248, # D
    57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265 # E
}


TEST_PATH = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\01. TEST_KJC.csv"
VAL_PATH  = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\02. VAL_KJC.csv"


# ================================================================
# 2. 전처리 함수 (Epitope Mutation Counting 강화)
# ================================================================

def count_epitope_muts(aa_str):
    """HA1의 주요 Epitope 부위에서 발생한 돌연변이 개수를 계산합니다."""
    if pd.isna(aa_str) or aa_str == '': return 0
    count = 0
    # Nextclade aaSubstitutions 형식 (예: HA1:S145N, HA1:A138S) 분석
    muts = str(aa_str).split(',')
    for m in muts:
        if 'HA1:' in m:
            match = re.search(r'(\d+)', m)
            if match and int(match.group(1)) in ALL_EPITOPE_SITES:
                count += 1
    return count

def load_and_process(path):
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path, sep=";", low_memory=False)
    
    # [수정] 연도 로직 강화: 2005~2025 범위 내의 숫자 4자리만 인정
    s = df["seqName"].astype(str)
    # 1. | 뒤에 오는 4자리 숫자 추출
    # 2. /YYYY| 형태 추출
    year_extracted = s.str.extract(r'\|(\d{4})')[0].fillna(s.str.extract(r'/(\d{4})\|')[0])
    df["year"] = pd.to_numeric(year_extracted, errors='coerce')
    
    # 유효 연도 필터링 (1904, 2099 등 이상치 제거)
    df = df[df["year"].between(2005, 2025)].copy()
    
    # Epitope 변수 계산
    df['epitope_count'] = df['aaSubstitutions'].apply(count_epitope_muts)
    
    results = []
    for (yr, cld), group in df.groupby(['year', 'clade']):
        results.append({
            'year': yr, 'clade': cld, 'n': len(group),
            'freq': len(group) / len(df[df['year']==yr]),
            'nonsyn_med': group['totalAminoacidSubstitutions'].median(),
            'novelty_med': group['privateAaMutations.totalUnlabeledSubstitutions'].median(),
            'epitope_mut_med': group['epitope_count'].median() # 신규 추가된 '질적' 지표
        })
    
    res_df = pd.DataFrame(results).sort_values(['clade', 'year'])
    res_df['freq_prev'] = res_df.groupby('clade')['freq'].shift(1).fillna(0)
    res_df['freq_delta'] = res_df['freq'] - res_df['freq_prev']
    return res_df

# ================================================================
# 3. 모델 학습 및 2026 예측 (Backtesting)
# ================================================================

try:
    val_df = load_and_process(VAL_PATH)
    test_df = load_and_process(TEST_PATH)
    full_df = pd.concat([val_df, test_df]).drop_duplicates(['year', 'clade']).sort_values(['year', 'clade'])

    # Target: 내년 최대 빈도 Clade 여부
    full_df['target'] = 0
    for yr in full_df['year'].unique():
        next_df = full_df[full_df['year'] == yr + 1]
        if not next_df.empty:
            best_clade = next_df.loc[next_df['freq'].idxmax(), 'clade']
            full_df.loc[(full_df['year'] == yr) & (full_df['clade'] == best_clade), 'target'] = 1

    # 피처 목록 (epitope_mut_med 포함)
    FEATURES = ['freq', 'freq_delta', 'nonsyn_med', 'novelty_med', 'epitope_mut_med']
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.8, C=0.4, random_state=42))
    ])

    print("\n" + "="*65)
    print("[9] Backtesting (Epitope & Fixed Timeline)")
    print("="*65)
    print(f"  Year   Actual CR  #1 Pred    #2 Pred    #3 Pred     H@1  H@3")
    print("  " + "-"*63)

    years = sorted(full_df['year'].unique())
    hits1, hits3, valid_years = 0, 0, 0
    
    for i, yr in enumerate(years):
        if i < 3: continue # 학습 데이터 확보
        train = full_df[full_df['year'] < yr]
        test = full_df[full_df['year'] == yr]
        if test.empty or train['target'].sum() == 0: continue
        
        model.fit(train[FEATURES], train['target'])
        test['prob'] = model.predict_proba(test[FEATURES])[:, 1]
        
        actual = full_df[(full_df['year'] == yr) & (full_df['target'] == 1)]['clade'].values
        actual = actual[0] if len(actual) > 0 else "unassigned"
        preds = test.sort_values('prob', ascending=False)['clade'].values[:3]
        
        h1 = "O" if actual == preds[0] else "X"
        h3 = "O" if actual in preds else "X"
        if h1 == "O": hits1 += 1
        if h3 == "O": hits3 += 1
        valid_years += 1
        
        mark = " <<<" if h1 == "O" else (" *" if h3 == "O" else "")
        print(f"  {int(yr):<6} {actual:<10} {preds[0]:<10} {preds[1] if len(preds)>1 else '-':<10} {preds[2] if len(preds)>2 else '-':<10}  {h1:>3}  {h3:>3} {mark}")

    # 2026 예측
    model.fit(full_df[FEATURES], full_df['target'])
    last_data = full_df[full_df['year'] == 2025].copy()
    if not last_data.empty:
        last_data['prob'] = model.predict_proba(last_data[FEATURES])[:, 1]
        p26 = last_data.sort_values('prob', ascending=False)['clade'].values[:3]
        print(f"  {'2026':<6} {'?':<10} {p26[0]:<10} {p26[1]:<10} {p26[2]:<10}   ?    ? (예측)")

    print("  " + "-"*63)
    print(f"  [Summary] Top-1: {hits1/valid_years:.1%}, Top-3: {hits3/valid_years:.1%}")

    # [10] 요약 정보 출력
    print("\n" + "="*65)
    print("[10] MODEL SUMMARY (Epitope Enhanced)")
    print("="*65)
    print(f"  Added Feature: epitope_mut_med (Wiley/Koel Sites)")
    print(f"  Biological Strategy: 'Quality over Quantity'")
    print(f"  Year Range:    2005 - 2025 (Fixed)")
    print(f"  Coefficient (Epitope): {model.named_steps['lr'].coef_[0][-1]:.4f}")

except Exception as e:
    print(f"\n[오류 발생]: {e}")