# Variate-Grouping iTransformer (VG-iT) Framework

## 1. 학술적 핵심 요약 (3-Line Summary)

*   **Core Idea:** 고차원 시계열 데이터의 변수를 토큰화하는 구조에서, 전역 소통을 **'학습 가능한 계층적 그룹 소통(Learnable Hierarchical Communication)'**으로 재구조화하여 노이즈와 연산량을 동시에 제어한다.
*   **Novelty:** 변수 간 상관관계를 모델이 스스로 학습하여 자동으로 그룹화하는 **최초의 학습 가능한 인버티드 계층 아키텍처(Learnable Inverted Hierarchy)**를 제안한다.
*   **Contribution:** 12-16GB VRAM 제약 하에서도 수천 개의 변수를 가진 대규모 시계열 데이터를 효율적으로 처리할 수 있는 **'Scalable Inverted Transformer'의 표준**을 제시한다.

---

## 2. 세부 설계 및 연구 배경

### 2.1 연구 배경 (Motivation)
- **Dense CD의 한계:** iTransformer와 같은 Channel-Dependent(CD) 모델은 모든 변수 간의 관계를 보려 하므로, 변수가 많아질수록(N > 500) 연산량이 기하급수적으로 늘어나고 상관관계가 낮은 변수들 사이의 노이즈가 학습을 방해한다.
- **산업 데이터의 특성:** 공장 센서, 교통 흐름 등 실제 산업 데이터는 변수 간의 계층적/지역적 상관관계가 뚜렷하며 노이즈가 많다. 이를 무조건 전역적으로 연결하는 것보다, 의미 있는 그룹 단위로 묶어 처리하는 것이 훨씬 견고(Robust)하다.

### 2.3 제안하는 해결책: VG-iT
- **Variate Grouping:** 학습 가능한 선형 투영(Learnable Projection)을 통해 각 변수를 대표하는 그룹으로 할당한다. 
- **Hierarchical Attention:** 
  1. **Intra-group:** 인접 센서/유사 특징들 사이의 미세한 패턴 분석.
  2. **Inter-group:** 그룹 간의 거시적인 흐름(Global Trend) 분석.
- **Efficiency:** 연산 복잡도를 $O(N^2)$에서 $O(G(N/G)^2 + G^2)$로 낮추어, 동일 VRAM 대비 훨씬 긴 Lookback Window를 확보할 수 있다.

---

## 3. 향후 로드맵
- [ ] **Learnable Grouping Layer 구현:** Soft-assignment 기반의 그룹화 메커니즘 개발.
- [ ] **노이즈 강건성(Noise Robustness) 검증:** 노이즈가 많은 Traffic/Industrial 데이터셋에서의 성능 입증.
- [ ] **확장성 벤치마크:** 12GB VRAM에서의 최대 수용 변수/시퀀스 길이 측정.
