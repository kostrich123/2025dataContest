import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. 데이터 정의 (중복된 2012년 중 하나 제거)
years = np.array([
    2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
    2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
    2022, 2023, 2024,2025
]).reshape(-1, 1)

counts = np.array([
    8, 6, 4, 8, 4, 12, 10, 9, 5, 6,
    7, 5, 5, 2, 8, 4, 4, 6, 3, 5,
    6, 5, 3,32
])

# 2. 선형 회귀 모델 훈련
model = LinearRegression()
model.fit(years, counts)

# 3. 예측할 연도 지정
future_years = np.array([2025, 2026, 2027, 2028]).reshape(-1, 1)
future_predictions = model.predict(future_years)

# 4. 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(years, counts, color='blue', label='Actual Data')  # 실제 데이터
plt.plot(years, model.predict(years), color='green', label='Linear Regression Line')  # 회귀선
plt.scatter(future_years, future_predictions, color='red', label='Predictions (2025-2028)')  # 예측값
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Linear Regression Forecast for 2025-2028')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 예측 결과 출력
for year, pred in zip(future_years.flatten(), future_predictions):
    print(f"{year}년 예측값: {pred:.2f}개")