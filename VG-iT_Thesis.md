# Variate-Grouping iTransformer (VG-iT): A Hierarchical Framework for Scalable Multivariate Time Series Forecasting

**Type:** Technical Report / Thesis Section Draft  
**Target:** Q1 Academic Journals & Master's Thesis  
**Status:** Monumental Edition (Deep-Dive)

---

## 1. Introduction

### 1.1 The Landscape of Multivariate Time Series Forecasting (MTSF)
The digital transformation of industrial systems has led to an explosion in the volume and complexity of time series data. Modern infrastructures rely on Multivariate Time Series Forecasting (MTSF) to predict future trends across thousands of interdependent variables. Unlike univariate approaches that model variables in isolation, MTSF captures the dynamic cross-correlations essential for system-wide optimization (Box et al., 2015). Historically, these two dimensions were modeled using linear regressions (VAR) or recurrent structures (LSTMs, GRUs). However, as data volume scaled, the field shifted toward Deep Learning (DL) to capture high-dimensional non-linearities (Wen et al., 2022).

### 1.2 The Evolution of Attention: From Temporal to Inverted Tokens
Early Transformer-based forecasting models (e.g., Informer (Zhou et al., 2021), Autoformer (Wu et al., 2021)) followed the natural language processing (NLP) paradigm, treating each time step as a token. While this allowed for powerful temporal modeling, it often relegated cross-variate correlations to secondary projections.

A significant paradigm shift occurred with the introduction of **PatchTST (Nie et al., 2023)** and **iTransformer (Liu et al., 2024)**. PatchTST advocated for **Channel-Independence (CI)** to avoid cross-channel noise contamination. Conversely, the **iTransformer** pioneered the **Inverted Paradigm**, treating each entire variate sequence as a single token. By applying self-attention to these variate-tokens, the model directly learns the correlation matrix between physical or logical entities. This approach has proven that capturing explicit channel dependencies is often superior to assuming independence in complex industrial systems where cross-variable interaction is high (Liu et al., 2024).

### 1.3 The Paradox of Scalability: The $O(N^2)$ Bottleneck
Despite its theoretical elegance, the inverted paradigm faces a computational paradox. The dense self-attention mechanism possesses a quadratic complexity of $O(N^2)$, where $N$ is the number of variates. In industrial scenarios involving thousands of sensors—such as the **Traffic** (862 variates) or **Electricity** (321 variates) datasets often used in SOTA benchmarks, and even larger real-world IoT grids—N can exceed 2,000 (Liu et al., 2024).
1.  **Memory Complexity**: A single attention map for $N=2,000$ with 8 heads consumes substantial VRAM, leading to Out-of-Memory (OOM) errors on consumer-grade and even workstation GPUs.
2.  **Attention Dilution (Rank Collapse)**: In high-dimensional regimes, attention weights often become overly distributed (diluted) across thousands of variables. This "noise floor" obscures critical cross-interactions—a phenomenon akin to **Rank Collapse** in deep Transformers, where the attention matrix tends toward a low-rank, uniform distribution as variables increase (Beltagy et al., 2020; Dong et al., 2021).
3.  **Real-world Applicability**: The requirement for massive enterprise hardware (e.g., NVIDIA H100 with 80GB HBM3 memory) to train such dense models creates a "scalability gap." While an H100 can process medium $N$, a single layer attention score map for $N=100,000$ (as found in global power grids) would require $\approx 320$ GB of peak memory (FP32), far exceeding the capacity of even high-end individual GPUs.

### 1.4 Proposal: Variate-Grouping iTransformer (VG-iT)
In this work, we present the **Variate-Grouping iTransformer (VG-iT)**, a hierarchical architecture designed to provide the benefits of full variate-dependency modeling while maintaining a tiered complexity suitable for high-dimensional data. Our core hypothesis is that high-dimensional variables are not randomly distributed but exhibit **"Systemic Locality"**—they form logical or physical clusters (e.g., sensors on the same equipment). 

VG-iT exploits this locality by:
1.  **Dividing $N$ variates into $G$ groups**, performing dense attention only within localized clusters.
2.  **Employing an Information Bottleneck (Pooling)** to summarize group-level macro-trends.
3.  **Facilitating Global Communication** through a sparse inter-group interaction layer.
The resulting complexity is $O(N^2/G + G^2)$, which is significantly more manageable for large $N$. We further introduce a **Salience-Aware Gated Integration Bridge** to ensure that while the architecture is hierarchical, individual variate details are never lost in the aggregation process.

### 1.5 Novelty and Original Contributions
To clearly distinguish the proposed VG-iT from existing works, we summarize its core contributions:
1.  **Hierarchical Variate Attention (Original Proposal)**: We bridge the gap between inverted transformers (iTransformer) and industrial scalability. By introducing a tiered grouping tier, we effectively trade off dense $O(N^2)$ global attention for a decoupled $O(N^2/G + G^2)$ interaction.
2.  **Salience-Aware Gated Integration Bridge (Original Proposal)**: We propose a learnable bridge that selectively fuses local "Salient" features with global "Consensus" trends, directly addressing the information loss found in traditional hierarchical aggregation.
3.  **Application of Information Bottleneck to Channel Denoising (Theoretic Synergy)**: We formalize the use of Mean Pooling not just as a reduction tool, but as a controlled Information Bottleneck (IB) (Tishby et al., 1999) to purify high-dimensional sensor noise (Feng et al., 2024).

---

## 2. Theoretical Background

### 2.1 The "Inversion" Rationale
The iTransformer (Liu et al., 2024) is built on the premise that variates in time series are analogous to words in a sentence. Just as a word's meaning is derived from its context within a sequence, a variate's future state is informed by its "context" relative to other variables. By inverting the dimensions, the model treats the temporal dimension (lookback window $L$) as the feature space ($d_{model}$) and the variate dimension as the token space. This ensures that the attention mechanism explicitly focuses on **Channel Correlations**.

### 2.2 Systemic Locality and Hierarchical Attention (Proposed)
In high-dimensional multivariate systems, the correlation matrix is typically sparse or block-diagonal. Sensors in the same equipment share fine-grained patterns (Intra-group), mientras que sensors in different areas share only macro-trends (Inter-group). 
VG-iT's hierarchical structure is mathematically inspired by the **Swin Transformer (Liu et al., 2021)**, but applied to the Channel dimension. We treat the Channel dimension as a sequence that can be partitioned into localized groups, reducing the receptive field of initial layers. 
> [!NOTE]
> **Ablation Needed**: The assumption of "Systemic Locality" in variate order or learned grouping needs empirical verification against random variable shuffling.

### 2.3 The Role of the Information Bottleneck (Proposed)
Simple hierarchical aggregation often leads to the loss of individual nuance. To mitigate this, we treat group representative generation as a controlled **Information Bottleneck (IB)** (Tishby et al., 1999). By using Fixed Mean Pooling, we act as a low-pass filter to smooth individual sensor noise while retaining group-level dynamics, a strategy similarly explored in extracting robust temporal dynamics (Feng et al., 2024).
> [!NOTE]
> **Empirical Validation Needed**: The "Denoising" effect of Mean Pooling vs. Max Pooling or Strided Convolution must be validated via ablation studies on noisy industrial datasets.

---

## 3. Methodology: Detailed Architecture

VG-iT transforms the vanilla iTransformer encoder into a tiered communication system.

### 3.1 Data Flow Overview
The input tensor $\mathcal{X} \in \mathbb{R}^{B \times L \times N}$ follows these transformations:
1.  **Inverted Embedding (iTransformer)**: $\mathbf{e}_n = \text{MLP}(x_{1:L, n}) \in \mathbb{R}^D$. Resulting in $\mathbf{H}_0 \in \mathbb{R}^{B \times N \times D}$.
2.  **Hierarchical Attention Layers (Proposed)**: $E$ layers of tiered interaction (Intra-group $\to$ Pooling $\to$ Inter-group $\to$ Gating).
3.  **Temporal Projection (iTransformer)**: The final variate-tokens are projected back to length $P$.

---

### 3.2 Stage 1: Variate Grouping and Intra-Group Attention (Proposed Construction)
This stage captures localized dynamics within clusters of variables.
1.  **Grouping Mechanism (Proposed Logic)**: Unlike PatchTST which groups along the *temporal* axis, VG-iT performs **Consecutive Channel-wise Windowing**. We view the $N$ variates as a sequence of tokens and partition them into $G$ non-overlapping, contiguous windows of size $M = \lceil N/G \rceil$. This assumes that adjacent variates in the input tensor share higher semantic locality (e.g., adjacent sensors in a physical grid). 
    - **Padding Logic**: If $N \pmod G \neq 0$, we calculate the required padding $P = (G \times M) - N$ and append $P$ zero-tokens to the **end of the variate dimension**. This ensures all groups have a uniform size $M$ for hardware-friendly parallel processing without shifting the relative indices of existing variates.
2.  **Reshaping Logic (Code-to-Sentence)**:
    - Input: $\mathbf{H} \in \mathbb{R}^{B \times N \times D}$

    - Pad $\mathbf{H} \to \mathbf{H}_{pad} \in \mathbb{R}^{B \times (G \times M) \times D}$

    - View $\to \mathbb{R}^{B \times G \times M \times D} \xrightarrow[Permute]{(2,0,1,3)} \mathbb{R}^{G \times B \times M \times D}$

    - Flatten $\to \mathbf{H}_{local} \in \mathbb{R}^{(G \cdot B) \times M \times D}$

3.  **Local Self-Attention**: Standard multi-head attention (Vaswani et al., 2017) is applied to the $M$ tokens within each group.

    $$\text{Local\_Out} = \text{Softmax}\left(\frac{Q_{local} K_{local}^T}{\sqrt{d}}\right) V_{local} \in \mathbb{R}^{(G \cdot B) \times M \times D}$$

    This captures dense inner-group correlations with $O(N^2/G \cdot D)$ complexity.

### 3.3 Stage 2: Inter-Group Global Communication (Proposed)
To ensure information flow between different clusters, we implement a global interaction tier.
1.  **Representative Generation (Pooling)**: Each group's output $[B \cdot G, M, D]$ is summarized into a single vector. We utilize **Fixed Mean Pooling**, treating it as an **Information Bottleneck (IB)** (Tishby et al., 1999) to act as a low-pass filter that smooths individual sensor noise.
    $$\mathbf{R} = \text{Mean}(\text{Local\_Out}, \text{dim}=M) \in \mathbb{R}^{B \times G \times D}$$
2.  **Global Self-Attention**: The $G$ representatives interact through a secondary attention layer.
    $$\text{Global\_Context} = \mathbf{Attention}(\mathbf{R}, \mathbf{R}, \mathbf{R}) \in \mathbb{R}^{B \times G \times D}$$
    This captures the macro-dependencies across the entire system by allowing cluster centroids to exchange information, ensuring global consistency.

### 3.4 Stage 3: Salience-Aware Gated Integration Bridge (Original Proposal)
To prevent individual variate nuances from being obscured by global context (the "Mean-field" problem), we introduce a learnable bridge.
1.  **Context Expansion**: The global context $[B, G, D]$ is expanded to match the original $N$ variates.
    $$\mathbf{H}_{global} = \text{Broadcast}(\text{Global\_Context}, M) \in \mathbb{R}^{B \times N \times D}$$
2.  **Salience Gating**: We compute a dynamic gate $\Gamma$ to evaluate the importance of local vs. global signals.
    $$\Gamma = \sigma(\text{MLP}([\mathbf{H}_{local}; \mathbf{H}_{global}])) \in \mathbb{R}^{B \times N \times D}$$
3.  **Adaptive Merge**: The final represention is a weighted sum based on variate salience.
    $$\mathbf{H}_{final} = \Gamma \odot \mathbf{H}_{global} + (1 - \Gamma) \odot \mathbf{H}_{local}$$
This mechanism ensures that variates with unique, "salient" local fluctuations preserve their integrity, while those following the general system trend accept the refined global context.

### 3.5 Forward Pass Comparison: iTransformer vs. VG-iT
To clarify the structural departure, we present the forward flow of the baseline followed by the proposed VG-iT in a unified, detailed mapping.

#### Table 1: Baseline iTransformer Forward Pass (Liu et al., 2024)
| Stage | Module | Tensor Shape | Logic |
| :--- | :--- | :--- | :--- |
| **0** | **Embedding** | `(B, N, D)` | Project Length $L$ to $D$ (Inversion) |
| **1** | **Global Attn** | `(B, N, D)` | Dense $N \times N$ Attention on all $N$ variates |
| **2** | **Residual** | `(B, N, D)` | $\mathbf{H} + \text{Attn}(\mathbf{H})$ |
| **3** | **Decoding** | `(B, N, P)` | Project $D$ to Prediction Horizon $P$ |

#### Table 2: Proposed VG-iT Forward Pass (Hierarchical Decomposition)
| Stage | Module | Tensor Shape (Shape) | Transformation & Ops | Semantic Intent | Baseline Diff |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Embedding** | `(B, N, D)` | `Invert(B, L, N)` | Variate-Token Formation | Identical |
| **1.1** | **Grouping** | `(B, G, M, D)` | `Window(N to GxM)` | Localized Cluster Prep | **Original Proposal** |
| **1.2** | **Intra-Attn**| `(B*G, M, D)` | `Self-Attn(M)` | Capture Inner Dynamics | Scaled down ($M \ll N$) |
| **2.1** | **Pooling** | `(B, G, D)` | `Mean(IB Method)` | Denoised Representatives | **Proposed** (Noise Filter) |
| **2.2** | **Inter-Attn**| `(B, G, D)` | `Self-Attn(G)` | Global System Communication | **Original Proposal** |
| **3.1** | **Expansion** | `(B, N, D)` | `Broadcast(M times)` | Redistribute Global Info | **Proposed Integration** |
| **3.2** | **Gating** | `(B, N, D)` | `Sigmoid(Salience)` | Adaptive Local-Global Merge | **Original Proposal** |
| **4** | **Decoding** | `(B, N, P)` | `Linear(D, P)` | Forecasting Projection | Identical |

---

## 4. Complexity and Theoretical Analysis

### 4.1 Formal Complexity Proof
Let $N$ be the number of variables, $G$ the number of groups, and $D$ the hidden dimension.
- **Vanilla iTransformer**: Complexity is $O(N^2 \cdot D)$. As $N \to 1,000$, $N^2 = 1,000,000$.
- **VG-iT**:
    1.  Intra-group Attention: $G \cdot (N/G)^2 \cdot D = \frac{N^2}{G} \cdot D$.
    2.  Inter-group Attention: $G^2 \cdot D$.
    3.  Total Complexity: $O\left(\left(\frac{N^2}{G} + G^2\right) \cdot D\right)$.

For $N=1,000$ and $G=32$, $M \approx 32$. Total complexity $\approx (1,000^2 / 32 + 32^2) \approx (31,250 + 1,024) = 32,274$.
This represents a **~31x reduction** in the computational burden of the attention layers.
Let $N=1,000, G=32$ (typical Industrial IoT config).
- **iTransformer**: $O(N^2) \approx 1,000,000$.
- **VG-iT**: $O(N^2/G + G^2) \approx (31,250 + 1,024) \approx 32,274$.
- **Saving**: $\approx 31\times$ computational reduction.

### 4.2 Mathematical Justification for Fixed Mean Pooling
In statistics, the **Law of Large Numbers** (Bernoulli, 1713) implies that the sample mean of a group of variables with shared underlying dynamics $\mu$ is a robust estimator that minimizes the variance of individual zero-mean noise $\epsilon$. By pooling variates within a group, we minimize the impact of individual sensor outliers $O(\epsilon)$, allowing the Global Attention layer to operate on a high-fidelity representation of the cluster's state.
$$\bar{X} = \frac{1}{M}\sum (X_i + \epsilon_i) \approx \mu$$
This effectively acts as a **Low-Pass Filter**, ensuring that only low-frequency group trends enter the $G^2$ global attention stage (Hyndman & Athanasopoulos, 2018).

---

## 5. References

Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*.

Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time series analysis: forecasting and control*. John Wiley & Sons.

Dong, Y., Cordonnier, J. B., & Loukas, A. (2021). Attention is not all you need: Pure attention loses rank at exponential rate with depth. *Proceedings of the International Conference on Machine Learning (ICML)*.

Feng, N., Lai, S., Yin, Z., Zhou, F., & Zhao, H. (2024). TimeSieve: Extracting temporal dynamics through information bottlenecks. *arXiv preprint arXiv:2401.07150*.

Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice*. OTexts.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278-2324.

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). iTransformer: Inverted transformers are effective for time series forecasting. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 10012-10022.

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *arXiv preprint physics/0004057*.

Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L. (2022). Transformers in time series: A survey. *arXiv preprint arXiv:2202.07125*.

Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. *Advances in Neural Information Processing Systems (NeurIPS)*.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, G., Ma, H., ... & Ye, J. (2021). Informer: Beyond efficient transformer for long-term time series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*.

---

## 6. [국문] 고차원 다변량 시계열 예측을 위한 계층적 채널 상관관계 학습 아키텍처: VG-iT

**본 섹션은 위 영문 기술 보고서의 내용을 1:1 대응하여 정밀하게 번역 및 보강한 것입니다.**

---

### 6.1 서론

#### 6.1.1 다변량 시계열 예측(MTSF)의 현황과 과제
산업 시스템의 디지털 전환은 시계열 데이터의 양과 복잡성을 폭발적으로 증가시켰습니다. 스마트 전력망, 글로벌 공급망, 대규모 클라우드 데이터 센터 등 현대 인프라는 수천 개의 상호 의존적인 변수들에 대한 미래 추세를 예측하기 위해 다변량 시계열 예측(MTSF)에 의존하고 있습니다. MTSF의 일차적인 과제는 변수가 시간에 따라 어떻게 변화하는지(시간적 종속성)와 변수들끼리 어떻게 상호작용하는지(변수 간 상관관계)를 동시에 포착하는 것입니다. 개별 변수를 독립적으로 모델링하는 단변량(Univariate) 방식과 달리, MTSF는 변수 간의 동적 교차 상관관계를 포착하여 시스템 전체의 최적화를 가능하게 합니다 (Box et al., 2015). 과거에는 이러한 모델링을 위해 선형 회귀(VAR)나 순환 구조(LSTM, GRU)가 사용되었으나, 데이터 규모가 커짐에 따라 고차원의 비선형성을 포착하기 위해 딥러닝(Deep Learning)으로 패러다임이 전환되었습니다 (Wen et al., 2022).

#### 6.1.2 어텐션의 진화: 시점 토큰에서 역전된(Inverted) 변수 토큰으로
초기 Transformer 기반 예측 모델(예: Informer (Zhou et al., 2021), Autoformer (Wu et al., 2021))은 자연어 처리(NLP) 패러다임을 따라 각 시점을 토큰으로 취급했습니다. 이는 강력한 시간적 모델링을 가능하게 했으나, 변수 간 상관관계를 선형 임베딩이나 고차원 투사(Projection)와 같은 부차적인 문제로 처리하는 경향이 있었습니다.

이러한 패러다임의 근본적 변화는 **PatchTST (Nie et al., 2023)** 와 **iTransformer (Liu et al., 2024)** 의 등장과 함께 일어났습니다. PatchTST는 채널 간 노이즈 혼입을 막기 위한 '채널 독립성(CI)'을 주장한 반면, **iTransformer** 는 전체 변수 시퀀스를 하나의 토큰으로 취급하는 **'역전된 패러다임(Inverted Paradigm)'** 을 개척했습니다. 변수-토큰에 자가주의(Self-attention)를 적용함으로써 모델이 변수들 간의 상관관계 행렬을 직접 학습하게 된 것입니다. 이 방식은 고도화된 물리 시스템에서 변수 간 종속성을 명시적으로 학습하는 것이 독립성을 가정하는 것보다 뛰어난 성능을 보임을 입증했습니다 (Liu et al., 2024).

#### 6.1.3 확장성의 역설: $O(N^2)$ 병목 현상
이론적 우수성에도 불구하고, 역전된 패러다임은 연산상의 치명적인 역설을 야기합니다. 밀집 자가주의 메커니즘은 변수 개수 $N$에 대해 $O(N^2)$의 복잡도를 가집니다. 최신 SOTA 벤치마크에서 사용되는 **Traffic** (862개 변수)이나 **Electricity** (321개 변수) 데이터셋뿐만 아니라, 실제 산업 현장의 IoT 그리드에서는 $N$이 2,000개를 쉽게 초과합니다 (Liu et al., 2024).
1.  **메모리 복잡도**: $N=2,000$일 때 8개의 헤드를 가진 단일 어텐션 층은 막대한 VRAM을 소모하며, 이는 보급형 GPU 환경에서 메모리 부족(OOM) 오류를 일으킵니다.
2.  **주의 희석 (Attention Dilution)과 랭크 붕괴(Rank Collapse)**: 고차원 환경에서 어텐션 가중치는 수천 개의 변수로 과도하게 분산(희석)되는 경향이 있습니다. 이는 깊은 Transformer 모델에서 변수 개수가 많아질수록 어텐션 행렬이 저차원의 균등 분포로 수렴하여 정보 구별력을 잃는 **랭크 붕괴(Rank Collapse)** 현상과 맥을 같이 합니다 (Beltagy et al., 2020; Dong et al., 2021).
3.  **실환경 적용성**: 이러한 고밀도 모델을 학습시키기 위해 80GB 이상의 HBM3 메모리를 갖춘 NVIDIA H100과 같은 엔터프라이즈급 하드웨어가 필수적이라는 점은 자원 제약이 있는 실제 산업 현장과의 큰 '확장성 공백'을 형성합니다. 예를 들어 FP32 기준으로 $N=100,000$ (글로벌 전력망 등)일 때 어텐션 맵 하나만으로 약 320GB의 VRAM이 필요하며, 이는 현존하는 단일 GPU의 한계를 아득히 초과합니다.

#### 6.1.4 제안 모델: VG-iT (Variate-Grouping iTransformer)
본 연구에서는 전역적 변수 종속성 모델링의 이점은 유지하면서도 고차원 데이터에 적합한 계층적 복잡도를 유지하는 **VG-iT (Variate-Grouping iTransformer)** 를 제안합니다. 우리의 핵심 가설은 고차원 변수들이 무작위로 분포하는 것이 아니라, 특정 장비의 센서들처럼 논리적/물리적 군집인 **'시스템적 지역성(Systemic Locality)'** 을 보인다는 것입니다.

VG-iT는 다음과 같은 방식으로 이 지역성을 활용합니다:
1.  **$N$개의 변수를 $G$개의 그룹으로 분할** 하여 국소적 군집 내에서만 밀집 어텐션을 수행합니다.
2.  **정보 병목 (Information Bottleneck, 풀링)** 을 활용하여 그룹 단위의 거시적 트렌드를 요약합니다.
3.  **희소한 그룹 간 상호작용 층** 을 통해 전역 통신을 촉진합니다.
이를 통해 연산 복잡도는 $O(N^2/G + G^2)$으로 감소하며, 이는 대규모 $N$ 환경에서 훨씬 관리 가능한 수준입니다. 또한, 계층 구조에서도 개별 변수의 세부 사항이 소실되지 않도록 **'Salience-Aware Gated Integration Bridge'** 를 도입했습니다.

#### 6.1.5 연구의 참신성 및 기여점 (Novelty & Contribution)
본 연구에서 제안하는 VG-iT의 핵심 기여점은 다음과 같습니다:
1.  **계층적 변수 상관관계 학습 (독자적 제안)**: 역전된 패러다임(iTransformer)의 장점은 유지하면서도 연산 복잡도를 기하급수적으로 낮추는 그룹화-대표값 교환 구조를 제안했습니다.
2.  **중요도 인지 게이트 통합 브리지 (독자적 제안)**: 계층적 모델의 고질적 문제인 정보 소실을 해결하기 위해, 로컬의 '특이값(Salient)'과 글로벌의 '일관성(Consensus)'을 적응적으로 병합하는 파라미터화된 게이트 구조를 설계했습니다.
3.  **정보 병목 기반의 채널 디노이징 (이론적 융합)**: 단순한 다운샘플링이었던 풀링을 **정보 병목(Information Bottleneck) 이론 (Tishby et al., 1999)** 관점에서 재해석하여, 고차원 센서 데이터의 노이즈를 효과적으로 정제하는 수리적 근거를 확립했습니다.

---

### 6.2 이론적 배경

#### 6.2.1 '역전(Inversion)'의 원리
iTransformer (Liu et al., 2024)는 시계열의 변수가 문장 속의 단어와 유사하다는 전제하에 설계되었습니다. 단어의 의미가 문맥에 의해 정의되듯, 변수의 미래 상태는 다른 변수들과의 관계(Context)에 의해 결정됩니다. 차원을 역전시킴으로써 모델은 시간 차원(Lookback Window $L$)을 특징 공간($d_{model}$)으로, 변수 차원을 토큰 공간으로 취급하게 됩니다. 이는 어텐션 메커니즘이 **채널 상관관계 학습(Channel Correlation Learning)** 에 직접 집중하도록 보장합니다.

#### 6.2.2 시스템적 지역성과 계층적 어텐션 (제안)
고차원 다변량 시스템에서 상관관계 행렬은 대개 희소(Sparse)하거나 블록 대각(Block-diagonal) 형태를 띱니다. 동일 장비 내의 센서들은 미세한 패턴(Intra-group)을 공유하는 반면, 멀리 떨어진 장비끼리는 거시적 트렌드(Inter-group)만을 공유한다는 가설입니다.
VG-iT의 계층 구조는 **Swin Transformer (Liu et al., 2021)** 에서 영감을 얻어 채널 차원에 적용되었습니다. 국소적 그룹 분할을 통해 어텐션의 수용 영역을 제한함으로써 가장 관련성 높은 종속성에 집중하게 합니다. 
> [!NOTE]
> **Ablation 필요**: 변수 순서의 무작위 셔플링 대비 '시스템적 지역성' 가설이 실제로 유효한지에 대한 실험적 검증이 향후 수행되어야 합니다.

#### 6.2.3 정보 병목(Information Bottleneck)의 역할 (제안)
단순한 계층적 합산은 정보 손실을 초래할 수 있습니다. 이를 방지하기 위해 우리는 그룹 대표값 생성을 통제된 **정보 병목(Information Bottleneck)** 으로 취급합니다 (Tishby et al., 1999). 평균 풀링은 개별 센서의 독립적 노이즈를 필터링하는 저주파 통과 필터(Low-pass filter) 역할을 하며, 전역 어텐션 층이 핵심적인 시스템 동역학에만 집중하도록 돕습니다. 이는 최근 시계열 데이터에서 유의미한 시간적 특징을 추출하기 위해 정보 병목 원리를 활용하는 연구들과 궤를 같이 합니다 (Feng et al., 2024).
> [!NOTE]
> **Ablation 필요**: 평균 풀링의 '디노이징' 효과가 다른 풀링 방식(Max, Strided Conv)보다 우수한지에 대한 정교한 비교 실험이 필요합니다.

---

### 6.3 방법론 상세: 아키텍처 구조

VG-iT는 바닐라 iTransformer 인코더를 계층적 통신 시스템으로 변환합니다.

#### 6.3.1 데이터 흐름 개요
1.  **역전된 임베딩 (iTransformer)**: 시계열 시퀀스를 변수별 토큰으로 변환합니다.
2.  **계층적 어텐션 층 (제안)**: Intra-group $\to$ Pooling $\to$ Inter-group $\to$ Gating 순으로 상호작용합니다.
3.  **시간적 투사 (iTransformer)**: 변수 토큰을 미래 예측 구간 $P$로 투사합니다.

#### 6.3.2 1단계: 변수 그룹화 및 그룹 내(Intra-Group) 어텐션 (제안 - 핵심 구조)
본 단계는 변수 군집 내의 국소적 동역학을 포착하며, 본 연구의 핵심적인 복잡도 절감 기법이 적용됩니다.
1.  **그룹화 메커니즘 (제안 로직)**: 시간축을 그룹화하는 PatchTST와 달리, VG-iT는 **'연속된 채널 기반 윈도잉(Consecutive Channel-wise Windowing)'** 을 수행합니다. $N$개의 변수를 토큰 시퀀스로 간주하고, 이를 중첩되지 않는 $G$개의 연속된 윈도우(크기 $M = \lceil N/G \rceil$)로 분할합니다. 이는 인접한 센서나 변수들이 물리적으로나 논리적으로 더 높은 상관관계를 가진다는 '시스템적 지역성' 가설에 기반합니다.
    - **패딩 상세 로직**: $N$이 $G$로 나누어떨어지지 않을 경우, 부족한 개수 $P = (G \times M) - N$만큼의 제로-토큰을 **변수 차원의 마지막(End of Sequence)에 추가**합니다. 이는 기존 변수들의 인덱스 순서를 유지하면서 모든 그룹이 동일한 크기 $M$을 갖게 하여, 최적의 GPU 병렬 연산을 보장하기 위함입니다.
2.  **재구성 로직 (Code-to-Sentence)**:

    - 입력: $\mathbf{H} \in \mathbb{R}^{B \times N \times D}$

    - 패딩: $\mathbf{H}_{pad} \in \mathbb{R}^{B \times (G \times M) \times D}$

    - 재구성: $\mathbf{H}_{pad} \xrightarrow[Reshape]{(B, G, M, D)} \xrightarrow[Permute]{(2, 0, 1, 3)} \mathbb{R}^{G \times B \times M \times D}$

    - 펼침: $\mathbf{H}_{local} \in \mathbb{R}^{(G \cdot B) \times M \times D}$

3.  **국소적 자가주의**: 표준 멀티헤드 어텐션 기술(Vaswani et al., 2017)을 적용합니다.

    $$\text{Local\_Out} = \text{Softmax}\left(\frac{Q_{local} K_{local}^T}{\sqrt{d}}\right) V_{local} \in \mathbb{R}^{(G \cdot B) \times M \times D}$$

    이를 통해 전체 변수 대비 훨씬 작은 $O(N^2/G)$ 복잡도로 정밀한 상관관계를 학습합니다.

#### 6.3.3 2단계: 그룹 간 전역 통신 (제안 - 계층 구조 확장)

국소 정보만으로는 포착할 수 없는 시스템 전체의 매크로 종속성을 유기적으로 연결합니다.

1.  **대표값 생성 (제안 - 정보 병목)**: 그룹 당 출력값을 평균으로 요약하여 그룹별 '대표 동역학' $\mathbf{R} \in \mathbb{R}^{B \times G \times D}$을 생성합니다. 이는 **Tishby et al. (1999)** 의 정보 병목 이론을 시계열에 적용하여 노이즈를 억제하는 과정입니다.

    $$\mathbf{R} = \text{Mean}(\text{Local\_Out}) \in \mathbb{R}^{B \times G \times D}$$

2.  **전역 자가주의**: 요약된 대표값 $\mathbf{R}$들끼리 두 번째 어텐션 연산을 수행합니다.

    $$\text{Global\_Context} = \mathbf{Attention}(\mathbf{R}, \mathbf{R}, \mathbf{R}) \in \mathbb{R}^{B \times G \times D}$$

    이는 각 군집의 '무게중심(Centroid)'들끼리 정보를 교환함으로써 시스템 전체의 전역적 일관성(Global Consistency)을 확보하는 본 모델만의 독창적인 설계입니다.

#### 6.3.4 3단계: Salience-Aware Gated Integration Bridge (독자적 제안 - 정보 복원)

계층 구조의 고질적 문제인 '평균화에 따른 개별 특징 소실'을 해결하는 핵심 장치입니다.

1.  **문맥 확장**: 전역 문맥을 다시 $M$배 확장하여 각 개별 변수의 원래 위치와 차원을 맞춥니다.

    $$\mathbf{H}_{global} = \text{Repeat}(\text{Global\_Context}, M) \in \mathbb{R}^{B \times N \times D}$$

2.  **변별적 게이팅 (Salience Gating - 신규 제안)**: 국소 특징($\mathbf{H}_{local}$)과 전역 문맥의 상대적 유의미함을 학습하는 시그모이드 게이트 $\Gamma$를 계산합니다.

    $$\Gamma = \sigma(\text{MLP}([\mathbf{H}_{local}; \mathbf{H}_{global}])) \in \mathbb{R}^{B \times N \times D}$$

3.  **적응형 병합**: 개별 특징이 강한(Salient) 변수는 로컬 신호를 보존하고($\Gamma \to 0$), 전체 흐름을 따르는 변수는 정제된 글로벌 문맥을 수용($\Gamma \to 1$)하도록 하여 정보의 정밀도를 유지합니다.

    $$\mathbf{H}_{final} = \Gamma \odot \mathbf{H}_{global} + (1 - \Gamma) \odot \mathbf{H}_{local}$$

#### 6.3.5 통합 순방향 흐름 비교: iTransformer vs. VG-iT
두 모델의 구조적 차이를 명확히 하기 위해, 전체 순방향 패스를 단계별로 비교한 통합 매핑 표입니다.

##### [표 1] 베이스라인 iTransformer 순방향 패스 (Liu et al., 2024)
| 단계 | 모듈 명칭 | 텐서 형상 | 핵심 로직 |
| :--- | :--- | :--- | :--- |
| **0** | **Embedding** | `(B, N, D)` | 시간을 특징($D$)으로 투사 (Inversion) |
| **1** | **Full Attn** | `(B, N, D)` | $N \times N$ 전역 자가주의 수행 (전체 변수 대상) |
| **2** | **Residual** | `(B, N, D)` | 잔차 연결 및 레이어 정규화 |
| **3** | **Decoding** | `(B, N, P)` | 미래 시점($P$)으로 최종 투사 |

##### [표 2] 제안 모델 VG-iT 순방향 패스 (계층적 분해 및 통합)
| 단계 | 모듈 | 텐서 형상 (Shape) | 변환 및 연산 | 의미론적 의도 | 베이스라인 대비 차이 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Embedding** | `(B, N, D)` | `Linear(L, D)` | 변수 시퀀스의 토큰화 | 동일 |
| **1.1** | **Grouping** | `(B, G, M, D)` | `Windowing(NxM)` | 국소 군집 형성 | **독자 제안** (지역성 확보) |
| **1.2** | **Intra-Attn**| `(B*G, M, D)` | `Self-Attn(M)` | 그룹 내 상관관계 포착 | 연산 효율 증대 ($M \ll N$) |
| **2.1** | **Bottleneck**| `(B, G, D)` | `Mean(IB-filter)` | 대표 동역학(R) 추출 | **제안** (노이즈 정제) |
| **2.2** | **Inter-Attn**| `(B, G, D)` | `Self-Attn(G)` | 거시적 시스템 학습 | **독자 제안** (클러스터 통신) |
| **3.1** | **Expansion** | `(B, N, D)` | `Broadcast(M-times)` | 전역 문맥의 재분배 | **제안** (통합 준비) |
| **3.2** | **Gating** | `(B, N, D)` | `Sigmoid(Gating)` | 로컬 vs 전역 정보 통합 | **독자 제안** (중요도 보존) |
| **4** | **Decoding** | `(B, N, P)` | `Linear(D, P)` | 시계열 미래 값 생성 | 동일 |

---

### 6.4 복잡도 및 이론적 분석

#### 6.4.1 공식적 복잡도 증명
- **기존 iTransformer**: $O(N^2 \cdot D)$. $N=1,000$일 때 $\approx 1,000,000$ 연산.
- **VG-iT**: $O\left(\left(\frac{N^2}{G} + G^2\right) \cdot D\right)$. $G=32$일 때 $\approx 32,274$ 연산.
- **결론**: 어텐션 연산량을 **약 31배 절감**하면서도 상관관계 학습 능력을 유지합니다.

#### 6.4.2 Fixed Mean Pooling의 수학적 정당성
통계학의 **대수의 법칙** (Bernoulli, 1713)에 따르면, 동일한 동역학 $\mu$를 공유하는 변수 그룹의 샘플 평균은 개별 노이즈 $\epsilon$의 분산을 최소화하는 강력한 추정치입니다.
$$\bar{X} = \frac{1}{M}\sum (X_i + \epsilon_i) \approx \mu$$
이는 공학적으로 **저주파 통과 필터(Low-pass Filter)** 역할을 수행하여, 고빈도 노이즈를 억제하고 전역 어텐션 층이 순수한 시스템 트렌드에 집중하도록 돕습니다 (Hyndman & Athanasopoulos, 2018).

---

## 7. Future Ablation Studies Needed (향후 검증 필요 사항)

본 보고서의 주장을 완성하기 위해 다음의 정교한 어블레이션(Ablation) 연구가 필수적으로 요구됩니다:

1.  **Systemic Locality Verification**: 변수 순서를 임의로 셔플링(Shuffle)했을 때와 사전 정의된 그룹화(Grouping)를 적용했을 때의 성능 차이를 통해 제안 구조의 정당성 확보.

2.  **Denoising Effect of Pooling**: Mean Pooling이 Max Pooling 또는 단순 Sampling 대비 실제 산업 데이터셋의 노이즈에 얼마나 강건(Robust)한지에 대한 정량적 분석.

3.  **Gate Salience Analysis**: Gating 브리지가 실제로 중요도(Salience)가 높은 변수에 대해 가중치를 다르게 부여하는지 분포 시각화 확인.

4.  **$O(G^2)$ Global Interaction Efficiency**: 그룹 간 통신이 부재할 때(Local attention only) 대비 글로벌 어텐션이 예측 오차율을 얼마나 개선하는지 확인.

5.  **Hyperparameter Sensitivity ($G$)**: 그룹 개수 $G$의 변화에 따른 예측 정확도와 VRAM 사용량 간의 트레이드오프(Trade-off) 분석.

6.  **Bridge Necessity (Gated Fusion vs. Additive Residual)**: 제안된 Gated Integration Bridge가 단순한 가산적 잔차 연결(Additive Residual Connection, $\mathbf{H}_{local} + \mathbf{H}_{global}$)보다 개별 변수의 고유 특성(Salience)을 보존하는 데 얼마나 더 기여하는지 검증합니다. 이는 계층적 통합 단계에서 비선형 게이트의 필요성을 입증하기 위한 정밀 검증 실험입니다.