# `model/base` 재활용 기준 실험 정리

현재 저장소에서 공통 실험 베이스는 아래 두 파일입니다.

- `model/base/selection.py`: `uniform_sampling(video_path, ...) -> torch.Tensor`
- `model/base/vlm.py`: `BaseVLM(model_id, frame_selector=..., ...)`

즉, 이 프로젝트의 기본 실험 단위는 아래 구조로 정리할 수 있습니다.

1. `frame_selector`가 영상을 읽고 `(T, H, W, C)` 형태의 프레임 텐서를 만든다.
2. `BaseVLM`이 해당 프레임 텐서와 프롬프트를 processor/model에 넣어 응답을 생성한다.
3. `config/*.yaml`에서 `frame_selection`과 `vlm`의 `_target_`만 바꿔 각 실험을 분기한다.

## 재활용 가능한 공통 파트

아래 요소는 대부분의 방식에서 그대로 재활용할 수 있습니다.

- 공통 실행 엔트리포인트: `model/invoke.py`
- 공통 VLM 로딩: `BaseVLM.build_vlm()`
- 공통 프롬프트 로딩: `load_prompt()`
- 공통 추론/출력 포맷: `BaseVLM.answer()`의 processor -> generate -> decode 흐름
- 공통 기본 설정 템플릿: `config/base.yaml`

정리하면, **프레임만 바꾸는 실험은 `frame_selection` 함수만 새로 만들면 되고**, **토큰 압축까지 건드리는 실험은 `BaseVLM`을 상속해 `answer()`를 확장하면 됩니다.**

## 1. Base Baseline

Base는 이미 구현되어 있는 기준선입니다.

- 프레임 선택: `model.base.uniform_sampling`
- VLM 실행: `model.base.BaseVLM`
- 설정 파일: `config/base.yaml`

실행 예시는 아래와 같습니다.

```bash
python -m model.invoke
```

이 구성은 새로운 실험을 만들 때 가장 먼저 복사해서 시작하는 템플릿으로 사용하면 됩니다.

## 2. AFS: `frame_selector`만 교체

AFS처럼 핵심 차이가 `frame selection`에만 있다면 `BaseVLM`은 그대로 쓰고, 선택 함수만 새로 구현하면 됩니다.

구현 위치 예시:

- `model/AFS/afs_selection.py`에 `adaptive_frame_sampling(video_path: str, ...) -> torch.Tensor` 추가

재활용 포인트:

- 그대로 사용: `BaseVLM`, `model/invoke.py`, prompt/query 로딩
- 새로 구현: 프레임 추출 및 선택 로직

가장 쉬운 시작 방법은 `uniform_sampling()`을 복사해 아래만 바꾸는 것입니다.

- 후보 프레임을 더 많이 뽑은 뒤 유사 프레임 제거
- optical flow / histogram / embedding distance 기반 중복 제거
- 마지막 반환 형식은 반드시 `torch.Tensor` 유지

설정 파일 예시는 아래 형태로 두면 됩니다.

```yaml
# config/afs.yaml
frame_selection:
  _target_: model.AFS.afs_selection.adaptive_frame_sampling
  _partial_: true
  num_frames: 8
  max_side: 720

vlm:
  _target_: model.base.BaseVLM
  model_id: "LanguageBind/Video-LLaVA-7B-hf"
  backend: "video_llava"
  dtype: "bf16"
  frame_selector: ${frame_selection}
  generation_kwargs:
    max_new_tokens: 256
    do_sample: false

invoke:
  video_path: "./data/fcN5HGqVzC0.mp4"
  prompt_file: "model/prompt.txt"
  query_file: "model/query.txt"
  print_config: false
```

실행:

```bash
# config-name 플래그를 통해 config 파일 선택 가능
python -m model.invoke --config-name afs
```

## 3. KTV: 1차는 frame-only, 2차는 token 단계 확장

KTV는 두 단계로 나눠 구현하면 가장 관리하기 쉽습니다.

### Step A. Keyframe selection만 먼저 검증

초기 실험에서는 AFS와 동일하게 `frame_selector`만 구현하고 `BaseVLM`을 그대로 사용합니다.

- 구현 위치: `model/KTV/ktv_selection.py`
- 역할: 중요 프레임만 골라 `torch.Tensor`로 반환

이 단계에서는 아래만 확인하면 됩니다.

- uniform baseline 대비 성능 차이
- 같은 `num_frames`에서 중복 감소 효과
- 프레임 선택 시간 증가 대비 정확도 개선 여부

### Step B. Key token selection까지 확장

토큰 레벨 pruning까지 넣고 싶다면 `BaseVLM`을 상속하는 편이 좋습니다.

예시 구조:

```python
class KTVVLM(BaseVLM):
    def answer(self, video_path: str, prompt: str, **frame_selector_kwargs):
        frames = self.frame_selector(video_path=video_path, **frame_selector_kwargs)
        inputs = self.processor(text=prompt, videos=frames, return_tensors="pt")
        # TODO: visual token importance 계산 / pruning
        # TODO: pruning된 입력으로 generate 호출
        ...
```

재활용 포인트:

- 그대로 사용: `build_vlm()`, dtype/backend 처리, prompt/query 로딩
- 새로 구현: token importance 계산, token pruning, 필요 시 hidden state hook

즉, **KTV의 frame-only 버전은 함수 추가만**, **full KTV는 `BaseVLM` 상속 추가**로 정리할 수 있습니다.

## 4. AIM: `BaseVLM` 상속으로 token compression 추가

AIM은 프레임 선택보다 **vision token compression / merging**이 핵심이므로, `frame_selector`보다는 `VLM inference path`를 확장하는 방식이 자연스럽습니다.

권장 구조:

- `frame_selection`: 우선 `model.base.uniform_sampling` 그대로 사용
- `vlm`: `model/AIM/aim_merge_pruning.py`에 `AIMVLM(BaseVLM)` 구현

AIM 쪽에서 주로 바꾸게 되는 부분은 아래입니다.

- processor 출력 이후 visual token sequence 접근
- token merging / pruning 적용
- 압축된 토큰으로 generate 또는 forward 수행

설정 예시는 아래처럼 시작하면 됩니다.

```yaml
# config/aim.yaml
frame_selection:
  _target_: model.base.uniform_sampling
  _partial_: true
  num_frames: 8
  max_side: 720

vlm:
  _target_: model.AIM.aim_merge_pruning.AIMVLM
  model_id: "LanguageBind/Video-LLaVA-7B-hf"
  backend: "video_llava"
  dtype: "bf16"
  frame_selector: ${frame_selection}
  generation_kwargs:
    max_new_tokens: 256
    do_sample: false

invoke:
  video_path: "./data/fcN5HGqVzC0.mp4"
  prompt_file: "model/prompt.txt"
  query_file: "model/query.txt"
  print_config: false
```

실행:

```bash
python -m model.invoke --config-name aim
```

## 5. 빠른 판단 기준

각 방식별로 어디를 수정하면 되는지 빠르게 정리하면 아래와 같습니다.

| 방식 | `frame_selection` 재사용 | `BaseVLM` 재사용 | 추가 구현 포인트 |
| --- | --- | --- | --- |
| Base | `uniform_sampling` 그대로 사용 | 그대로 사용 | 없음 |
| AFS | 새 함수로 교체 | 그대로 사용 | 프레임 중복 제거 |
| KTV-frame | 새 함수로 교체 | 그대로 사용 | keyframe 선택 |
| KTV-full | 새 함수 사용 가능 | 상속 후 확장 | key token selection |
| AIM | base 또는 AFS 재사용 가능 | 상속 후 확장 | token merging / pruning |

## 6. 추천 작업 순서

실험을 추가할 때는 아래 순서로 진행하면 가장 안전합니다.

1. `config/base.yaml`을 복사해 새 config 파일을 만든다.
2. 프레임 실험이면 `frame_selection._target_`만 새 함수로 바꾼다.
3. 토큰 실험이면 `vlm._target_`을 `BaseVLM`의 서브클래스로 바꾼다.
4. 실행은 `python model/invoke.py --config-name ...` 형태로 통일한다.
5. 비교 지표는 최소한 `selection time`, `end-to-end latency`, `task performance`를 함께 기록한다.

현재 코드 기준으로는 **Base는 바로 실행 가능하고**, **AFS/KTV/AIM은 비어 있는 파일에 위 구조대로 구현을 채워 넣는 방식**으로 확장하면 됩니다.
