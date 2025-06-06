#%% md
# # review_data.csv 를 전처리 하는 코드
#%%
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('data/Coupang-reviews-homeplanet/review_data.csv')
#%%
df.head(5)
#%%
# 헤드라인과 리뷰내용이 모두 결측값인 데이터와 아닌 데이터로 분리
df_complete_text = df.dropna(subset=['헤드라인', '리뷰내용'], how='all')  # 둘 다 NaN인 경우 제거
df_non_text = df[df['헤드라인'].isnull() & df['리뷰내용'].isnull()]  # 둘 다 NaN인 경우만 선택
#%%
df_complete_text.head()
#%%
df_complete_text.describe()
#%%
# df_complete_text 데이터 시각화
import matplotlib.pyplot as plt

# 한글 폰트
plt.rcParams['font.family'] = 'AppleGothic'  # macOS

# 평점 분포 시각화
plt.figure(figsize=(10, 6))
df_complete_text['평점'].value_counts().sort_index().plot(kind='bar')
plt.title('평점 분포')
plt.xlabel('평점')
plt.ylabel('리뷰 개수')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
#%%
df_non_text.head()
#%%
df_non_text.describe()
#%%
# df_non_text 데이터 시각화
plt.figure(figsize=(10, 6))
df_non_text['평점'].value_counts().sort_index().plot(kind='bar')
plt.title('평점 분포 (헤드라인과 리뷰내용이 모두 결측값인 경우)')
plt.xlabel('평점')
plt.ylabel('리뷰 개수')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
#%%
# 두 데이터 프레임을 다른 CSV 파일로 저장
df_complete_text.to_csv('data/Coupang-reviews-homeplanet/review_data_complete_text.csv', index=False)
df_non_text.to_csv('data/Coupang-reviews-homeplanet/review_data_non_text.csv', index=False)
#%%
df_complete_text.head()