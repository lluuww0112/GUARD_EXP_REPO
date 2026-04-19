# Frame Selection 수식 모델링

# 모델 수식

## 기본 구조

$$
z_k = \text{Score}(f^{(k)}\ |\ q,\ S_{k-1})\\
s_{k+1} = g(z_k, \text{budget state})
$$

- k: 결정 step
- $f^{(k)}$: step k에서의 frame
- q: query
- $S_{k-1}$ : step k-1까지 선택된 frame index의 set
- $s_{k+1}$: k+1 step에서 사용할 stride

**budget 상태와 점수를 통해, stride를 결정** 

## 점수 산출 수식

$$
z_k = \alpha r_k\ +\ \beta i_k(\delta_i + (1-\delta_i)r_k) +\gamma n_k(\delta_n + (1-\delta_n)r_k)\\ \text{s.t.}\ \alpha+\beta+\gamma=1
$$

- $r_k$: Semantic query relevance of frame (쿼리와 연관성)
- $i_k$: Frame intrinsic importance (프레임 자체의 중요도)
- $n_k$: novelty (프레임 커버리지)
- $\delta_i$: $i_k$와 $r_k$를 연결 (쿼리 연관 우선)
- $\delta_n$: $n_k$와 $r_k$를 연결 (쿼리 연관 우선)

**$z_k$가 threshold를 넘으면 해당 frame을 선택**

**$α, β, γ,$ $δ_i$,$δ_n$ 는 hyper parameter** 

### 세부 Term

> **Semantic query relevance of frame**
> 

$$
r_k  =\frac{1+ \cos(x^{(k)},u_q)}{2}
$$

- $x^{(k)}$: step k에서의 frame embedding
- $u_q$: query embedding

> **Frame intrinsic importance**
> 

$$
i_k = \frac{1-\cos(x^{(k)},\bar{x}_{k-1})}{2}\\
\text{where }\space \bar{x}_{k} =\rho \bar{x}_{k-1}  + (1-\rho){x}^{(k)} 
$$

- $x^{(k)}$: step k에서의 frame embedding
- $\bar{x}_{k-1}$: k-1 step까지의 frame embedding 평균

현재까지 흐름과 다른 정도에 따라 frame 자체의 중요도를 판단 (쿼리에 관련 없이 그냥 frame 자체만 보는 항)

**ρ는 hyper parameter**

> **novelty**
> 

$$
n_k = 1-p_k \\ 
\ \\
\text{where }\ p_k=\max_{j\in S_{k-1}} \left[\frac{1+\cos(x^{(k)},x_j)}{2}\text{exp}\left(-\frac{|m_k-j|}{\tau_{\text{temp}}} \right) \right]
\\ \ \\
\begin{cases}
p_k = 0 \text{ if}\space S_{k-1} = \emptyset \\
n_k = 1 \text{ if}\space S_{k-1} = \emptyset
\end{cases}
$$

- $p_t$: step k에서의 frame에 대한 novelty penalty
- $\underset{j\in S_{k-1}}{\max}\frac{1+\cos(x^{(k)},x_j)}{2}$:  $S_{k-1}$의 frame 중 step k에서의 frame에 대한 가장 높은 유사도
- $\tau_{\text{temp}}$: 시간이 지남에 따라 얼마나 빨리 중복으로 보는 정도를 약하게 만들지의 threshold
- $m_k$: k번째 결정 step에서 관측한  frame의 **index**
- $\text{exp}(-\frac{|m_k-j|}{\tau_{\text{temp}}})$: 가장 높은 유사도를 가진 frame의 시점과 현재 $m_k$가 가까울수록 억제
- $n_k$: 1- $p_k$(novelty penalty)

가까운 구간에서의 비슷한 frame은 중복으로 보고, 먼 구간에서의 비슷한 frame은 중복으로 덜 보게 하는 항 (너무 hard한 비교일 수 있음, soft한 비교도 생각해봐야 할 듯) 

$\tau_{\text{temp}}$가 작을수록 감쇠가 빠르다. 즉, 조금만 멀어져도 중복으로 잘 안 본다

**$\tau_{\text{temp}}$는 hyper parameter**

## Score에 기반한 Stride 결정

$$
s_{k+1}^\text{score} = s_\text{min}+(s_\text{max}-s_\text{min})(1-\bar{z}_k)^p\\
\text{where }\space \bar{z}_{k} = \lambda\bar{z}_{k-1}+(1-\lambda)z_k
$$

- $s_{k+1}^\text{score}$: **점수에 기반한** k+1 step에서의 stride
- $\bar{z}_k$: k step까지의 평균 score (점수 smoothing을 수행한 값)
- $(s_\text{max}-s_\text{min})(1-\bar{z}_k)^p$: $\bar{z}_k$이 클 때, 중요한 구간이므로 stride를 줄임

최소 stride $s_{\text{min}}$에 얼마의 stride를 더할지 결정하는 수식 

**λ, p는 hyper parameter**

## Budget에 기반한 Stride 결정

$$
s_{k+1}^\text{budget} = \max \left(1,\  \lfloor \frac{N_\text{rem}}{\max(K_\text{rem},1)}\rfloor \right)

$$

- $s_{k+1}^\text{budget}$ : budget에 기반한 k+1 step에서의 stride
- $N_\text{rem}$:  남은 frame 수
- $K_\text{rem}$: 남은 선택 가능한 frame 수(budget)

stride가 너무 커져서 budget을 다 못 채우는 경우를 위한 규제항 

## 최종 stride 결정

$$
s_{k+1}=\text{clip}\left(\min \{ s_{k+1}^{\text{score}},\ s_{k+1}^{\text{budget}}\},\ s_\text{min},\ s_{\text{max}}\right)
$$

$s_{k+1}^\text{score}$ , $s_{k+1}^\text{budget}$ 에서 더 작은 stride를 선택하고, stride의 bound 안에 위치시키도록 clip 연산 수행 

**$s_\text{min},\ s_{\text{max}}$은 hyper parameter**

## 계산 안전성을 위한 장치

### 최대 공백 제한 정책

$$
m_k-m_{\text{last}} \ge T_\text{max}\\
a_k = 1[z_k \ge \theta_k \lor m_k-m_\text{last} \ge T_\text{max}]\\
m_\text{last} = \begin{cases}
m_k\ \ \ \ \ \ \ \ \ \ \text{if}  \  \text{a}_k =1\\
m_{\text{last}}\ \ \ \ \ \ \ \text{if} \ \text{a}_k =0
\end{cases}
$$

- $m_k$: 현재 보고 있는 frame의 **index**
- $m_\text{last}$: 마지막으로 frame select를 수행한 frame index (frame을 뽑지 않은 $a_k=0$에서 유지)
- $T_\text{max}$: 공백 하한
- $a_k$: step k에서의 action ( $a_k=1$일 때, frame select)

Frame selection을 적어도 $T_\text{max}$이상으로 비우지 않도록 한다.  강제로라도 뽑게해서, stride가 너무 커져서 못 보는 구간이 너무 길지 않도록 한다. 

**$T_\text{max}$ 는 값을 어떻게 정해야 할지 못 정함** 

**$T_\text{max}$는 hyper parameter**

### 동적 Threshold

$$
\theta_k = \theta_0 - \eta[b_k]\\
\text{where }\space b_k = \frac{Km_k}{T} - |S_{k-1}|,\ [b_k]=\max(b_k,0)
$$

- $\theta_k$: step k에서의 threshold 값
- $\theta_0$: default threshold 값
- $\frac{Km_k}{T}$: 현재시점 $m_k$까지 선택되었어야할 frame의 기댓값
- $|S_{k-1}|$: 현재 step 직전까지 실제로 선택된 frame 수
- $\eta$: 뽑았어야할 frame과 실제로 뽑힌 frame 수 차이 반영 계수
- $[b_k]$: $b_k$가 양수일 때만 동작

$b_k$가 양수라면 뽑혔어야하는 값보다 덜 뽑혔다는 의미로 해석.
따라서, $\theta$를 낮춰서 frame 선택 문턱을 낮춰 더 잘 뽑히게 한다. 

여기서 $\theta$ 는 novelty 수식에 있는 threshold가 아니라, $z_k$에 대한 threshold.

**$\eta$는 hyper parameter**

# 전체 파이프라인

1. 점수($z_k$) 계산 
2. $\theta_k$ 계산
3. $a_k = 1[z_k \ge \theta_k \lor m_k-m_\text{last} \ge T_\text{max}]$ 결정 
4. 선택된 frame set $S_k$ 갱신, $m_\text{last}$ 갱신
5. 다음 stride ($s_{k+1}$) 계산
6. stride만큼 이동 

**선택되지 않은 경우에도 $s_{k+1}$은 최종 stride 결정 수식을 통해 결정된다.** 

$$
a_k \in \{0,1\}\\
a_k = 1[z_k \ge \theta_k \lor m_k-m_\text{last} \ge T_\text{max}] \\
S_k = \begin{cases} 
S_{k-1} \cup \{m_k\}\ \ \ \ \ \text{ if }\ a_k=1 \\
S_{k-1}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \text{ if }\ a_k = 0
\end{cases}\\
s_{k+1}=\text{clip}\left(\min \{ s_{k+1}^{\text{score}},\ s_{k+1}^{\text{budget}}\},\ s_\text{min},\ s_{\text{max}}\right)\\
m_{k+1} = m_k + s_{k+1}
 
$$

아직도 Naive한 버전이라서 smoothing 추가가 필요한지 검증해봐야 함 

hard한 부분이 있어서 경계에서 값이 잘못 판정될 가능성 있음 

> **스무딩 추가 후보**
> 
1. action 결정 
    
    $a_k = 1[z_k \ge \theta_k \lor m_k-m_\text{last} \ge T_\text{max}]$ 
    
    $\theta_k$랑 차이가 작은 $z_k$는 노이즈에 민감할 수 있어 위험함
    
2. novelty penalty 
    
    $p_k=\max_{j\in S_{k-1}} \left[\frac{1+\cos(x^{(k)},x_j)}{2}\text{exp}\left(-\frac{|m_k-j|}{\tau_{\text{temp}}} \right) \right]$
    
    max라서 가장 높은 frame 하나랑 비교하는 게 위험함 
    
3. dynamic threshold 
    
    $\theta_k = \theta_0 - \eta[b_k]\ \ \ \ \ \ \ \
    \text{where }\space b_k = \frac{Km_k}{T} - |S_{k-1}|$
    
    정수 단위로 $S_{k-1}$이 변해서 이산적으로 움직이는 게 걸림