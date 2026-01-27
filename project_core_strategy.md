# iTransformer-TTT Project Core: Philosophy & Strategy

## 1. Lessons from CTSF-V3 (Post-Mortem)
CTSF-V3 프로젝트의 실패에서 배운 교훈들을 새로운 프로젝트의 **"Iron Rules(철칙)"**로 삼습니다.

### 🚫 The "Don't Do" List (실패 원인)
1.  **Over-Engineering Trap:**
    *   *실수:* `RR_SCCPPP`처럼 너무 많은 모듈(Dual Head, Gating, Reg-Loss, Aux-Loss)을 덕지덕지 붙임.
    *   *결과:* 디버깅 불가능, VRAM 폭발, 학습 속도 저하.
    *   *교훈:* **"Single File, Single Logic."** 베이스라인(iTransformer) 코드에 침습적으로 들어가지 않고, 래핑(Wrapping)하거나 상속(Inheritance)하는 형태로 구현한다.
2.  **Premature Optimization:**
    *   *실수:* `AMP`, `Lazy Aggregation`, `Fused Kernel` 등을 로직 검증보다 먼저 적용하려 함.
    *   *교훈:* **"Make it work, then make it fast."** 일단 돌아가는 TTT 로직을 만들고, 속도 최적화는 나중에 한다.
3.  **Complexity without Purpose:**
    *   *실수:* 명확한 이유 없이 "좋아 보이는 것(Cross Attention 등)"을 섞어 씀.
    *   *교훈:* **"Why?"에 대한 답이 없는 모듈은 추가하지 않는다.**

---

## 2. Why is SOTA TTT Complex? (TTT 논문의 복잡성 분석) 
(TTT: SPECIALIZATION AFTER GENERALIZATION: TOWARDS UNDERSTANDING TEST-TIME TRAINING IN FOUNDATION MODELS)

최신 논문(NeurIPS 2024 등)들이 TTT를 복잡하게 구현하는 데는 이유가 있습니다. 우리는 이를 이해하고, **"Simple TTT"**로 어떻게 이를 대체할지 정의해야 합니다.

### A. TTT가 복잡한 이유 (The Challenges)
1.  **Gradient Step is Slow:** 인퍼런스 때마다 역전파(Backprop)를 하면 속도가 매우 느려집니다. 논문들은 이를 가속화하기 위해 Meta-learning을 쓰거나 전용 모듈을 만듭니다.
2.  **Proxy Task Mismatch:** 테스트 시점에는 정답($Y_{future}$)이 없습니다. 대신 "$X_{history}$를 복원해라" 같은 **Proxy Task(대리 과제)**를 줍니다.
    *   *위험:* "과거를 잘 맞추는 가중치"가 "미래를 잘 맞추는 가중치"와 다를 수 있습니다. (Negative Transfer).
    *   *논문의 해결책:* 이를 방지하기 위한 복잡한 Loss 설계나 전용 Head 구조.
3.  **Catastrophic Forgetting:** 적응하다가 기존에 학습한 일반적인 지식을 까먹어서 성능이 오히려 떨어지는 현상.

### B. 우리의 해결책 (Our "Simple" Strategy)
우리는 복잡한 모듈 없이 다음과 같이 논리적으로 방어합니다.
1.  **Speed:** "우리는 Real-time System이 아니다." 성능(Accuracy)을 위해 약간의 Latency는 감수한다는 전제. (단, 모델이 가벼운 iTransformer라 괜찮음).
2.  **Task Alignment:** 시계열 데이터는 **"Local Stationarity(국소적 정상성)"**이 강합니다.
    *   *가설:* "방금 전 96 step을 잘 맞추도록 수정된 모델은, 바로 다음 96 step도 잘 맞출 것이다."
    *   *구현:* Proxy Task로 **"Lookback Window 자체에 대한 Forecasting"**을 사용합니다.
3.  **Stability:** Learning Rate를 매우 작게 잡고, 1~3 step만 업데이트하여 Forgetting을 방지합니다.

---

## 3. iTransformer-TTT Implementation Core
### ⚙️ Architecture Design
**Base Model:** `iTransformer` (Official Implementation)
*   **특징:** Time series point들을 Embedding하여 Variate Token으로 만듦. 전체 Lookback Window를 한 번에 처리.
*   **TTT 포인트:** Layer Normalization 파라미터나 마지막 Linear Head만 업데이트하는 것이 일반적이지만, iTransformer는 구조가 단순하므로 **전체 파라미터(Full Fine-tuning)**를 살짝 건드려도 됩니다.

### 🧪 The TTT Logic (Algorithm)
`evaluate()` 함수 내부 로직:

1.  **Input:** Test Batch $X_{in} \in (B, L, D)$ ($L$: Lookback length)
2.  **Evaluation Mode:** `model.eval()` (기본)
3.  **TTT Step (Adaptation):**
    *   `model.train()` (일시적 전환)
    *   **Self-Supervised Data Generation:**
        *   입력 $X_{in}$을 둘로 쪼갭니다.
        *   $X_{sub\_in} = X_{in}[:, :-P, :]$ (과거의 과거)
        *   $Y_{sub\_target} = X_{in}[:, -P:, :]$ (과거의 최근) -> **우리가 맞혀야 할 정답 역할**
    *   **Forward & Loss:**
        *   $\hat{Y}_{sub} = \text{Model}(X_{sub\_in})$
        *   $L_{TTT} = \text{MSE}(\hat{Y}_{sub}, Y_{sub\_target})$
    *   **Update:**
        *   $\theta' = \theta - \eta \cdot \nabla_{\theta} L_{TTT}$ (1~3 steps)
4.  **Final Prediction:**
    *   `model.eval()`
    *   업데이트된 $\theta'$로 실제 예측: $\hat{Y}_{future} = \text{Model}(X_{in})$
5.  **Reset:**
    *   다음 배치를 위해 $\theta'$를 원래 $\theta$로 되돌림 (선택 사항, 논문에서는 보통 Reset함).

---

## 4. Useful Remnants from CTSF-V3 (구조 재활용)
CTSF-V3에서 실패했지만, TTT-iTransformer에서는 유용할 수 있는 기술들입니다.

### A. Frequency Domain Aux Loss (from D-Mamba/DLinear attempts)
*   **아이디어:** TTT 단계에서 단순히 MSE(시간 도메인)만 맞추게 하면 노이즈에 과적합될 수 있습니다.
*   **적용:** $L_{TTT}$를 계산할 때, **FFT Loss**를 추가하여 "주파수 특성"을 유지하도록 강제하면 TTT가 더 안정적일 것입니다.
    *   *Why?* iTransformer가 Global Receptive Field를 가지므로 주파수 특성을 보존하는 것이 유리함.

### B. Instance Normalization / RevIN (이미 iTransformer에 있음)
*   **주의:** TTT를 할 때 Normalization 통계량(Mean/Std)을 업데이트할지 말지 결정해야 합니다.
*   **제안:** 통계량은 그대로 두고, **모델의 가중치(Attention Map 등)**만 업데이트하여 "분포 변화에 따른 패턴 변화"를 학습하게 합니다.

---

## 5. Development Roadmap
1.  **Phase 1 (Clean Setup):** Official iTransformer Repo 복제 및 실행 환경 구축 (CTSF-V3 잔재 제거).
2.  **Phase 2 (Establish Baseline):** 순정 iTransformer로 ETTh1/Traffic 데이터셋 1 Epoch 벤치마킹.
3.  **Phase 3 (Implement TTT Wrapper):** `exp/ttt_wrapper.py` 같은 별도 모듈로 TTT 로직 구현. 모델 코드를 직접 건드리지 않음.
4.  **Phase 4 (Validation):** "TTT 적용 시 성능 향상" 확인. (Learning Rate, Step 수 튜닝)

# iTransformer-TTT Project Core: Philosophy & Strategy

## 1. Lessons from CTSF-V3 (Post-Mortem)
CTSF-V3 프로젝트의 실패에서 배운 교훈들을 새로운 프로젝트의 **"Iron Rules(철칙)"**로 삼습니다.

### 🚫 The "Don't Do" List (실패 원인)
1.  **Over-Engineering Trap:**
    *   *실수:* `RR_SCCPPP`처럼 너무 많은 모듈(Dual Head, Gating, Reg-Loss, Aux-Loss)을 덕지덕지 붙임.
    *   *결과:* 디버깅 불가능, VRAM 폭발, 학습 속도 저하.
    *   *교훈:* **"Single File, Single Logic."** 베이스라인(iTransformer) 코드에 침습적으로 들어가지 않고, 래핑(Wrapping)하거나 상속(Inheritance)하는 형태로 구현한다.
2.  **Premature Optimization:**
    *   *실수:* `AMP`, `Lazy Aggregation`, `Fused Kernel` 등을 로직 검증보다 먼저 적용하려 함.
    *   *교훈:* **"Make it work, then make it fast."** 일단 돌아가는 TTT 로직을 만들고, 속도 최적화는 나중에 한다.
3.  **Complexity without Purpose:**
    *   *실수:* 명확한 이유 없이 "좋아 보이는 것(Cross Attention 등)"을 섞어 씀.
    *   *교훈:* **"Why?"에 대한 답이 없는 모듈은 추가하지 않는다.**

---

## 2. Methodology Comparison: SOTA vs. Ours
사용자님이 언급하신 논문(NeurIPS 2024 등)과 우리의 접근 방식을 명확히 대조합니다.

### A. SOTA 논문의 방식 (The "Complex" Way)
*   **구조:** 기존 모델 + **별도의 TTT 전용 모듈(Learner)** + **Meta-learning Network**.
*   **학습법:** "TTT를 잘하기 위한 방법"을 또 학습시킴(Meta-train).
*   **목적:** 수천 개의 시계열을 엄청나게 빠르게 처리해야 하는 산업용 (Latency < 10ms).
*   **단점:** 구현이 극도로 어렵고, 모델 구조를 뜯어고쳐야 함.

### B. 우리의 방식 (The "Simple" Way)
*   **구조:** **순정 iTransformer** (변경 없음).
*   **학습법:** 인퍼런스 루프 안에서 **표준 역전파(Standard Backprop)** 수행.
*   **목적:** "정확도(Accuracy)" 극대화. Latency가 100~200ms 걸려도 상관없는 고정밀 예측.
*   **장점:** `Wrapper` 함수 하나로 구현 가능. 어떤 모델(iTransformer, PatchTST, DLinear)에도 즉시 적용 가능.

---

## 3. Concrete Implementation Logic (The Algorithm)
이 부분이 우리가 구현할 **핵심 엔진(Core Engine)**입니다.

### 🎯 Objective
입력 시계열 $X_{in}$ (길이 $L$)이 들어왔을 때, 이 데이터의 **최신 분포(Distribution)**에 모델을 적응시킨 후 미래를 예측한다.

### 📝 Step-by-Step Algorithm
이 로직은 `exp/ttt_wrapper.py`에 구현될 것입니다.

**Input:**
*   Test Batch $B_{test} \in \mathbb{R}^{B \times L \times D}$ ($L=96$)
*   Pre-trained Model $M_{\theta}$
*   TTT Hyperparams: $\eta$ (Learning Rate), $K$ (Iter Steps)

**Process (for each batch):**
1.  **Backup State:**
    *   현재 모델 파라미터 $\theta$를 CPU나 메모리에 백업합니다.
    *   `state_dict_m = copy.deepcopy(model.state_dict())`

2.  **Self-Supervised Data Generation (Proxy Task):**
    *   입력 $X_{in}$ (길이 96)을 "문제"와 "정답"으로 쪼갭니다.
    *   **Case 1 (Lookback Prediction):**
        *   입력: $X_{sub} = X_{in}[:, :48, :]$ (앞 절반)
        *   정답: $Y_{sub} = X_{in}[:, 48:, :]$ (뒤 절반)
        *   *의도:* "과거의 절반을 보고 현재의 절반을 맞추는 연습을 하면, 현재를 보고 미래를 맞추는 것도 잘할 것이다."

3.  **TTT Loop (Adaptation):**
    *   Switch to Train Mode: `model.train()`
    *   **For $k = 1$ to $K$:** (보통 1~3회)
        *   Prediction: $\hat{Y}_{sub} = M_{\theta}(X_{sub})$
        *   Loss: $\mathcal{L}_{TTT} = \text{MSE}(\hat{Y}_{sub}, Y_{sub})$
        *   **[Core]** Gradient Update:
            *   $\theta \leftarrow \theta - \eta \cdot \nabla_{\theta} \mathcal{L}_{TTT}$
            *   *중요:* 이때 `LayerNorm` 파라미터만 업데이트할지, `All Parameters`를 업데이트할지 실험 포인트로 잡음.

4.  **Inference (Testing):**
    *   Switch to Eval Mode: `model.eval()`
    *   적응된 파라미터 $\theta'$를 사용하여 실제 미래 예측.
    *   $\hat{Y}_{future} = M_{\theta'}(X_{in})$ (전체 입력 사용)

5.  **Restore State (Reset):**
    *   다음 배치가 "이전 배치의 정보"에 오염되지 않도록 파라미터를 원상복구합니다.
    *   `model.load_state_dict(state_dict_m)`
    *   *옵션:* 시계열이 이어지는 경우(Online), 복구하지 않고 계속 이어서(Continual) 학습할 수도 있음 (실험 포인트).

---

## 4. Useful Components from CTSF-V3 (Legacy Integration)
실패한 프로젝트에서 건질 만한 부품들입니다. TTT 로직 안에 **신중하게** 통합합니다.

### A. Frequency Domain Loss (Safe to use)
*   **용도:** TTT Loop 단계에서 MSE Loss 보조용.
*   **로직:** $\mathcal{L}_{TTT} = \text{MSE}(y, \hat{y}) + \lambda \cdot \text{MSE}(\text{FFT}(y), \text{FFT}(\hat{y}))$
*   **효과:** 적은 데이터(Batch=1)로 학습할 때, 모델이 고주파 노이즈에 과적합되어 그래프가 튀는 현상을 막아줌. (Smoothing 효과)

### B. Dynamic Learning Rate (Scheduler)
*   **용도:** TTT Step마다 LR을 줄여가며 미세 조정.
*   **로직:** 첫 스텝엔 과감하게($10^{-3}$), 둘째 스텝엔 살살($10^{-4}$).

---

## 5. Development Roadmap
1.  **Stage 1: Clean Base**
    *   `Time-Series-Library` (Official) 기반으로 최소한의 환경 구축.
    *   `iTransformer` 기본 성능 재현 (Replication).
2.  **Stage 2: TTT Injection**
    *   모델 코드 수정 없이, `exp_main.py`의 `test()` 함수만 오버라이딩하여 위 **Algorithm** 구현.
3.  **Stage 3: Verification**
    *   `ETTh1` 데이터셋에서 TTT 적용 전/후 Error 감소 확인.
    *   변수: Update Step 수, Learning Rate, Partial vs Full Update.
