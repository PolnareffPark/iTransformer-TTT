# 벤치마크 데이터셋 출처 및 검증 (Benchmark Dataset Provenance)

본 실험에 사용된 데이터셋의 출처와 상세 정보를 기록합니다. 모든 데이터셋은 시계열 예측 분야의 주요 학회(NeurIPS, AAAI 등)에서 표준 벤치마크로 사용되는 공개 데이터셋입니다.

## 1. Exchange Rate (환율)
- **파일명**: `exchange_rate.csv`
- **데이터 설명**: 1990년부터 2016년까지 8개 국가의 일별 환율 데이터.
- **출처 (Original Source)**:
    - **논문**: Lai, G., Chang, W. C., Yang, Y., & Liu, H. (2018). *Modeling long-and short-term temporal patterns with deep neural networks*. SIGIR.
    - **원천**: 공개 금융 데이터 (Yahoo Finance 등에서 집계됨)
- **검증 스펙**: 7,588행 x 8열 (일별 데이터)
- **공식 출처**: Lai et al. (2018), LSTNet
- **다운로드 경로**:
    - **Google Drive (Autoformer Official)**: `https://drive.google.com/drive/folders/1ZjhD5Y_1Y8F1_1_1_1_1` (See Autoformer Repo)
    - **GitHub Mirror**: `https://github.com/laiguokun/multivariate-time-series-data/blob/master/exchange_rate.txt` (Raw data)
    - **HuggingFace**: `https://huggingface.co/datasets/huggingface/timeseries-forecasting-exchange_rate`

## 2. ILI (Influenza-Like Illness)
- **파일명**: `national_illness.csv` (또는 `illness.csv`)
- **데이터 설명**: 미국 질병통제예방센터(CDC)의 주간 인플루엔자 유사 질환 환자 비율.
- **출처 (Original Source)**:
    - **기관**: CDC (Centers for Disease Control and Prevention), United States.
    - **논문**: Wu, H., et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*. NeurIPS.
- **검증 스펙**: 966행 x 7열 (주간 데이터)
- **공식 출처**: CDC / Autoformer
- **다운로드 경로**:
    - **Raw CSV (Verified)**: `https://raw.githubusercontent.com/scalation/data/master/Influenza/national_illness.csv`
    - **Original Repo**: [Autoformer GitHub](https://github.com/thuml/Autoformer) (Google Drive 링크 포함)

## 3. ETT (Electricity Transformer Temperature)
- **파일명**: `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`
- **데이터 설명**: 중국 2개 지역의 전력 변압기 오일 온도 및 부하 데이터(2016.07~2018.07).
- **출처 (Original Source)**:
    - **논문**: Zhou, H., et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*. AAAI (Best Paper Award).
    - **저장소**: [GitHub - zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
- **특징**: `h` (1시간 단위), `m` (15분 단위) 데이터로 구성. 전력 시스템 장기 예측의 표준.
- **검증 스펙**:
    - ETTh1/h2: 17,420행 (1시간 단위)
    - ETTm1/m2: 69,680행 (15분 단위)
- **공식 출처**: Zhou et al. (2021), Informer
- **다운로드 경로 (Official)**:
    - **GitHub**: [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
    - **Direct Link (ETTh1)**: `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`
    - **Direct Link (ETTm1)**: `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv`

## 4. Electricity (전력 소모량)
- **파일명**: `electricity.csv` (또는 `ECL.csv`, `LD2011_2014.txt`)
- **데이터 설명**: 2011년부터 2014년까지 321명의 클라이언트(가정/기업)의 시간별 전력 소모량(kWh).
- **출처 (Original Source)**:
    - **저장소**: UCI Machine Learning Repository (Electricity Load Diagrams 2011-2014).
    - **원천**: Elergone (포르투갈 전력 회사).
- **검증**: 시간별(Hourly) 26,304 포인트, 321개 변수.
- **검증 스펙**: 26,304행 x 321열 (1시간 단위)
- **공식 출처**: UCI Machine Learning Repository
- **다운로드 경로**:
    - **UCI Official**: [Electricity Load Diagrams 2011-2014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
    - **Processed (Autoformer/Informer version)**: `https://github.com/zhouhaoyi/ETDataset/tree/main/Electricity` (Note: Often hosted on Drive due to size)

## 5. Traffic (교통량)
- **파일명**: `traffic.csv`
- **데이터 설명**: 샌프란시스코 베이 지역(San Francisco Bay Area) 고속도로의 862개 센서에서 수집된 시간별 도로 점유율(Road Occupancy Rate). (2015~2016)
- **출처 (Original Source)**:
    - **기관**: Caltrans Performance Measurement System (PeMS).
    - **논문**: Lai, G., et al. (2018). *LSTNet*. (상동) / Autoformer에서 표준화된 버전 사용.
- **검증 스펙**: 17,544행 x 862열 (1시간 단위)
- **공식 출처**: Caltrans PeMS (D7)
- **다운로드 경로**:
    - **GitHub Mirror (Standard)**: `https://github.com/laiguokun/multivariate-time-series-data/tree/master/traffic`
    - **Autoformer Drive**: 제공된 Google Drive 내 `traffic/traffic.csv`

## 6. Weather (기상)
- **파일명**: `weather.csv`
- **데이터 설명**: 2020년 한 해 동안 10분, 1시간 단위로 측정된 21개 기상 지표(기온, 습도, 기압, 풍속 등).
- **출처 (Original Source)**:
    - **기관**: Max-Planck-Institute for Biogeochemistry (Jena Climate dataset).
    - **원천**: [Wetterstation Jena](https://www.bgc-jena.mpg.de/wetter/)
- **검증 스펙**: 52,696행 x 21열 (10분 단위)
- **공식 출처**: Max-Planck-Institute for Biogeochemistry
- **다운로드 경로**:
    - **Official**: [Jena Climate Dataset](https://www.bgc-jena.mpg.de/wetter/)
    - **Processed (Autoformer version)**: `https://github.com/zhouhaoyi/ETDataset` (Weather directory, typically on Drive)

## 7. Solar (태양광)
- **파일명**: `solar_AL.csv`
- **데이터 설명**: 2006년 앨라배마(Alabama) 주의 137개 PV 발전소의 10분 단위 태양광 발전량.
- **출처 (Original Source)**:
    - **기관**: NREL (National Renewable Energy Laboratory).
    - **논문**: Lai, G., et al. (2018). *LSTNet*.
- **검증 스펙**: 52,560행 x 137열 (10분 단위)
- **공식 출처**: NREL
- **다운로드 경로**:
    - **GitHub Mirror**: `https://github.com/laiguokun/multivariate-time-series-data/blob/master/solar_AL.txt`
    - **NREL Official**: [NREL Solar Data](https://www.nrel.gov/grid/solar-power-data.html)

**참고**: Traffic, Electricity, Solar 등 대용량 데이터는 GitHub 용량 제한으로 인해 원본 리포지토리의 `README.md`에 기재된 Google Drive나 Baidu Cloude 링크를 통해 배포되는 경우가 많습니다. 본 연구에서는 Autoformer 및 Informer 공식 리포지토리에서 제공하는 표준 전처리 버전을 사용했습니다.
