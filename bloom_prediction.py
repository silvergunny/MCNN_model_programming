import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 데이터 로드 (기상 데이터)
# 파일 인코딩에 주의하세요 (cp949)
weather_df = pd.read_csv('OBS_ASOS_DD_20260428161758.csv', encoding='cp949')

# 2. 기상 데이터 전처리
weather_df['일시'] = pd.to_datetime(weather_df['일시']) 
weather_df['연도'] = weather_df['일시'].dt.year
weather_df['월'] = weather_df['일시'].dt.month

# 결측치 처리: 강수량은 0, 나머지는 선형 보간
weather_df['일강수량(mm)'] = weather_df['일강수량(mm)'].fillna(0)
weather_df = weather_df.interpolate(method='linear')

# 3. 피처 엔지니어링 (개화에 영향을 주는 봄철 데이터 추출)
# 보통 1~3월 사이의 평균 기온과 누적 강수량이 개화에 큰 영향을 미칩니다.
spring_weather = weather_df[weather_df['월'].isin([1, 2, 3])].groupby('연도').agg({
    '평균기온(°C)': 'mean',
    '최저기온(°C)': 'min',
    '최고기온(°C)': 'max',
    '일강수량(mm)': 'sum',
    '합계 일조시간(hr)': 'sum'
}).reset_index()

# 4. 개화 데이터 로드 (사용자가 가진 2021~2025 데이터를 이 형식으로 합쳐주세요)
# 예시: bloom_df = pd.read_csv('bloom_date.csv')
# 여기서는 예시를 위해 가상 데이터를 생성합니다.
bloom_data = {
    '연도': [2021, 2022, 2023, 2024, 2025],
    '개화일수': [85, 88, 82, 90, 84]  # 1월 1일부터 흐른 일수
}
bloom_df = pd.DataFrame(bloom_data)

# 데이터 결합
final_df = pd.merge(spring_weather, bloom_df, on='연도')

# 5. 모델 학습 준비
X = final_df.drop(['연도', '개화일수'], axis=1)
y = final_df['개화일수']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 하이퍼파라미터 튜닝 (GridSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt'],
    'max_depth': [5, 10, 15, 20]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# 최적 모델 선택
best_model = grid_search.best_estimator_

# 7. 예측 및 결과 확인
y_pred = best_model.predict(X_test)
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"평균 절대 오차(MAE): {mean_absolute_error(y_test, y_pred):.2f}일")

# 8. 변수 중요도 해석
importances = pd.Series(best_model.feature_importances_, index=X.columns)
print("\n[변수 중요도]")
print(importances.sort_values(ascending=False))
