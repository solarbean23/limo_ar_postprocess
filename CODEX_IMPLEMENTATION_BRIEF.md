# LIMO AR Post-processing — Codex Implementation Brief

## 0. 목적

ROS 2 bag 파일(`.db3` + `metadata.yaml`)과 DJI Osmo 영상(`.lrf` 또는 `.mp4`)을 입력으로 받아, LIMO 실험 영상을 후처리 AR 영상으로 렌더링한다.

1차 구현은 **불필요하게 큰 프레임워크를 만들지 않고**, 아래 두 가지 출력 모드만 안정적으로 지원하는 것을 목표로 한다.

- `ar_only`: 원본 영상 위에 AR overlay만 입힌 영상
- `wide_with_map`: 원본 영상 왼쪽 + 오른쪽에 RViz-like 2D map을 붙인 wide 영상

텍스트 dashboard, 복잡한 GUI, 웹 UI, 실시간 처리, ROS runtime 의존성은 1차 구현 범위에서 제외한다.

---

## 1. 입력 파일

기본 입력은 다음과 같다.

```text
input_video.lrf 또는 input_video.mp4
rosbag.db3
metadata.yaml
```

현재 예시 데이터 기준:

```text
Video: DJI_20000710120811_0008_D.LRF
Bag:   limo_mission_20260525_214255_0.db3
Meta:  metadata.yaml
```

DJI `.lrf`는 일반적으로 MP4 계열 컨테이너로 OpenCV/ffmpeg에서 읽을 수 있다.  
나중에 고화질 `.mp4`를 사용할 수 있도록 video path만 바꾸면 같은 코드가 동작해야 한다.

---

## 2. 출력 모드

### 2.1 `ar_only`

원본 영상 크기와 비율을 그대로 유지한다.

예:

```text
Input video: 1280 x 720
Output:      1280 x 720
```

영상 위에 다음 AR 요소를 그린다.

- Robot ground ring / ellipse
- Robot label
- Target marker
- Fire marker
- Base area
- Fire suppress shrinking animation
- Rescue target count marker

이 모드는 RViz-like map이 필요 없는 경우를 위한 것이다.

---

### 2.2 `wide_with_map`

원본 영상은 왼쪽에 그대로 두고, 오른쪽에 영상 높이와 같은 변 길이를 가진 정사각형 2D map을 붙인다.

예:

```text
Input video: 1280 x 720
Map:          720 x 720
Output:      2000 x 720
```

고화질 mp4가 1920 x 1080이면:

```text
Input video: 1920 x 1080
Map:         1080 x 1080
Output:      3000 x 1080
```

구조:

```text
┌──────────────────────────────┬──────────────┐
│                              │              │
│       Original Video         │  RViz-like   │
│       + AR Overlay           │  2D Map      │
│                              │              │
└──────────────────────────────┴──────────────┘
```

우측 하단 패널은 1차 구현 범위에서 제외한다.  
필요하면 나중에 별도 `with_dashboard` 모드로 추가한다.

---

## 3. ROS bag에서 사용할 토픽

현재 예시 bag 기준 주요 토픽은 다음과 같다.

```text
/Rescue_Limo_1/pose_world      geometry_msgs/msg/PoseStamped
/Fire_Limo_1/pose_world        geometry_msgs/msg/PoseStamped
/Fire_Limo_2/pose_world        geometry_msgs/msg/PoseStamped

/world/base/pose               geometry_msgs/msg/PoseStamped
/world/fire/state              std_msgs/msg/String
/world/target/state            std_msgs/msg/String
/agent_broadcast               std_msgs/msg/String
```

1차 구현에서 필수 토픽:

```text
/Rescue_Limo_1/pose_world
/Fire_Limo_1/pose_world
/Fire_Limo_2/pose_world
/world/base/pose
/world/fire/state
/world/target/state
```

`/agent_broadcast`는 1차에서는 선택사항이다. task/phase dashboard를 넣을 때 사용한다.

---

## 4. World 좌표계

예시 실험은 약 4m x 4m 영역이다.

기본 map bounds:

```yaml
world:
  x_min: -2.0
  x_max:  2.0
  y_min: -2.0
  y_max:  2.0
```

중앙은 `(0, 0)`이다.

RViz-like map에서는 화면 좌표를 다음처럼 변환한다.

```text
screen_x = (x - x_min) / (x_max - x_min) * map_width
screen_y = (y_max - y) / (y_max - y_min) * map_height
```

즉, world 좌표 기준으로:

```text
좌측 상단: (-2,  2)
우측 상단: ( 2,  2)
좌측 하단: (-2, -2)
우측 하단: ( 2, -2)
```

현재 예시 bag에서 확인된 주요 object 위치는 대략 다음과 같다.

```text
Base center: (-1.0,  1.0)

Target_1:    ( 1.0,  1.0)
Target_2:    ( 1.0, -1.0)

Fire_1:      ( 0.5,  1.5)
Fire_2:      ( 0.5, -0.5)
Fire_3:      ( 1.2,  1.2)   # 중간 생성
Fire_4:      (-1.0, -1.0)   # 중간 생성
```

현재 bag 기준 fire는 초기에 2개이고, 중간에 추가 fire가 생성된다.

---

## 5. Base 영역

1차 기본값은 base pose `(-1, 1)` 주변의 사분면으로 잡는다.

```yaml
base_area:
  x_min: -2.0
  x_max:  0.0
  y_min:  0.0
  y_max:  2.0
```

Base는 반투명 회색 영역으로 표시한다.

Base 안에 들어온 robot 수에 따라 밝기를 조금씩 증가시킨다.

```text
0 robots in base: dark transparent gray
1 robot  in base: slightly brighter
2 robots in base: brighter
3 robots in base: brightest
```

---

## 6. 시나리오 설명

전체 mission phase는 다음과 같다.

### Phase 1: `AllTargetCompleted`

- Rescue Limo: `CompleteTargetAfterSafe`
- Fire Limo: `EscortLeader`

동작:
- Rescue Limo가 target을 향해 이동한다.
- 근처에 fire가 있으면 Rescue Limo는 wait한다.
- 가까운 Fire Limo가 해당 fire를 suppress한다.
- suppress는 약 2초 소요된다.
- fire가 사라지면 Rescue Limo가 target으로 다시 이동한다.
- target을 구조하면 해당 target은 map/video overlay에서 사라진다.
- 구조한 target 개수만큼 Rescue Limo 근처에 작은 노란색 사각형을 표시한다.

### Phase 2: `AllFireSuppressed`

- Rescue Limo: `IsInBase`
- Fire Limo: `SuppressFireAfterAssign`

동작:
- 모든 target 구조 후 Rescue Limo는 base로 복귀한다.
- Fire Limo들은 남은 fire를 suppress한다.
- 중간에 추가 fire가 생기면 가까운 Fire Limo가 처리한다.

### Phase 3: `AllInBase`

- Rescue Limo: `IsInBase`
- Fire Limo: `IsInBase`

동작:
- 모든 fire가 suppress되면 Fire Limo들도 base로 복귀한다.

1차 구현에서 phase text를 화면에 표시할 필요는 없다.  
다만 추후 dashboard를 넣기 쉽게 timeline 내부에서 target/fire 완료 이벤트는 계산해두면 좋다.

---

## 7. 시각화 규칙

### 7.1 실제 video AR overlay

영상 위에는 world 좌표를 homography로 video pixel 좌표에 투영해서 그린다.

필수 요소:

| 요소 | 시각화 |
|---|---|
| Base | 반투명 회색 polygon |
| Target | 노란색 사각형 |
| Fire | 옅은 빨간 원 |
| Rescue Limo | 흰색 ground ring/ellipse + label |
| Fire Limo | 빨간 ground ring/ellipse + label |
| Rescue target count | Rescue Limo 아래/옆의 작은 노란 사각형들 |

Robot ring은 카메라가 비스듬한 시점이므로 완전한 원보다 **지면에 붙은 타원형**으로 보이게 그리는 것이 자연스럽다.  
다만 1차 구현에서는 homography로 얻은 robot 중심점 근처에 ellipse를 그리고, yaw에 맞춰 회전시키는 정도로 충분하다.

### 7.2 RViz-like 2D map

오른쪽 map에는 top-view 형태로 그린다.

필수 요소:

| 요소 | 시각화 |
|---|---|
| Map bounds | 4m x 4m square |
| Base | 반투명 회색 rectangle |
| Target | 노란색 square |
| Fire | 옅은 빨간 circle |
| Rescue Limo | 흰색 작은 car body + `R` |
| Fire Limo | 빨간 작은 car body + `F` |
| Wheels | 검정색 작은 wheel 4개 |
| Heading | car yaw 반영 |

Robot은 단순한 top-view car icon으로 충분하다.  
과도한 3D 모델링은 하지 않는다.

---

## 8. Fire suppress animation

Bag에는 보통 fire가 active → inactive로 바뀐 상태만 기록되어 있다.  
시각적으로는 inactive 시점 직전 2초 동안 fire 원을 줄인다.

예:

```text
Fire_1 inactive time = 22.64 sec
suppress animation = 20.64 sec ~ 22.64 sec
radius scale = 1.0 → 0.0
```

config:

```yaml
visual:
  fire_suppress_duration_sec: 2.0
```

Fire가 active이면 정상 radius로 표시한다.  
Fire가 suppress animation 구간이면 radius를 선형 감소시킨다.  
그 이후에는 표시하지 않는다.

---

## 9. Target rescue visualization

Target이 active → inactive 또는 status changed로 완료되면 map/video에서 제거한다.

Rescue Limo가 구조한 target 수를 표시하기 위해 Rescue Limo 근처에 작은 노란색 square를 누적해서 표시한다.

예:

```text
0 targets rescued: no small square
1 target rescued:  one yellow square
2 targets rescued: two yellow squares
```

1차 구현에서는 정확히 어떤 target을 들고 있는지보다 완료 개수를 보여주면 충분하다.

---

## 10. Video ↔ Map robot connector lines

`wide_with_map` 모드에서 같은 robot이 video와 map에서 어떤 것인지 직관적으로 보이도록 연결선을 그린다.

각 frame마다:

```text
video robot position -> map robot position
```

을 얇은 반투명 선으로 연결한다.

권장:
- Rescue Limo: white / light gray semi-transparent line
- Fire Limo: red semi-transparent line
- 너무 두껍게 하지 말 것
- 라벨과 겹치면 보기 지저분하므로 alpha를 낮게 유지할 것
- config에서 켜고 끌 수 있게 할 것

config:

```yaml
visual:
  draw_connectors: true
```

`ar_only` 모드에서는 connector line을 그리지 않는다.

---

## 11. Homography calibration

실제 video overlay에는 world 좌표를 video pixel 좌표로 변환하는 homography가 필요하다.

필요 파일 예:

```yaml
homography:
  world_points:
    - [-2.0,  2.0]
    - [ 2.0,  2.0]
    - [ 2.0, -2.0]
    - [-2.0, -2.0]
  image_points:
    - [u1, v1]
    - [u2, v2]
    - [u3, v3]
    - [u4, v4]
```

OpenCV:

```python
H, _ = cv2.findHomography(world_points_2d, image_points_2d)
```

World point `(x, y)`를 image point `(u, v)`로 변환:

```python
p = H @ np.array([x, y, 1.0])
u = p[0] / p[2]
v = p[1] / p[2]
```

1차 구현에는 간단한 calibration script를 둔다.

예:

```bash
python -m limo_ar.calibrate \
  --video data/input.lrf \
  --world-points configs/world_points.yaml \
  --output configs/homography.yaml
```

script는 첫 프레임 또는 지정한 timestamp frame을 띄우고, 사용자가 대응되는 image point를 클릭하도록 한다.

다만 GUI script가 환경 문제를 일으킬 수 있으므로, 수동으로 `homography.yaml`을 작성해도 렌더러가 동작해야 한다.

---

## 12. Time synchronization

영상 길이와 bag 길이가 다를 수 있다.

현재 예시:
- LRF video: 약 73.9 sec
- rosbag: 약 89.0 sec

따라서 config에서 time offset을 조정할 수 있어야 한다.

```yaml
sync:
  time_offset_sec: 0.0
```

렌더링 시:

```python
bag_t = video_t + time_offset_sec
```

Preview를 짧게 렌더링하면서 offset을 조정한다.

---

## 13. 권장 폴더 구조

최소 구조로 시작한다.

```text
limo_ar_postprocess/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ example_limo.yaml
├─ limo_ar/
│  ├─ __init__.py
│  ├─ bag_reader.py
│  ├─ timeline.py
│  ├─ calibration.py
│  ├─ projection.py
│  ├─ drawing.py
│  └─ render.py
└─ outputs/
   └─ .gitkeep
```

불필요한 초기 파일은 만들지 않는다.

### 각 파일 역할

#### `bag_reader.py`
- `.db3`와 `metadata.yaml` 읽기
- 필요한 topic 추출
- pose, fire state, target state를 raw timeline으로 변환
- ROS가 설치되어 있지 않아도 동작하는 방식 권장
- 가능하면 `rosbags` 라이브러리 사용
- 너무 복잡하면 현재 메시지 타입에 대한 direct parser fallback 구현

#### `timeline.py`
- 특정 시간 `t`에서 현재 상태 반환
- robot pose interpolation
- fire active/inactive 상태
- target active/inactive 상태
- suppress animation timing
- rescued target count 계산

#### `calibration.py`
- homography yaml load/save
- world → video pixel 변환
- calibration helper

#### `projection.py`
- world → map pixel 변환
- world → video pixel 변환 wrapper
- output layout 좌표 변환

#### `drawing.py`
- OpenCV drawing utilities
- base, target, fire, robot ring, top-view car, connector line 등을 그림
- renderer에서 직접 복잡한 drawing code가 늘어나지 않게 함

#### `render.py`
- CLI entry point
- video frame 읽기
- timeline state 가져오기
- AR overlay 적용
- 필요하면 RViz-like map 생성
- output video 저장

---

## 14. 권장 config 예시

```yaml
input:
  video_path: data/DJI_20000710120811_0008_D.LRF
  bag_path: data/limo_mission_20260525_214255_0.db3
  metadata_path: data/metadata.yaml
  homography_path: configs/homography.yaml

output:
  path: outputs/limo_ar_wide_with_map.mp4

render:
  mode: wide_with_map        # ar_only | wide_with_map
  start_sec: 0.0
  end_sec: null
  max_frames: null
  fps: null                  # null이면 input video fps 사용
  codec: mp4v

sync:
  time_offset_sec: 0.0

world:
  x_min: -2.0
  x_max:  2.0
  y_min: -2.0
  y_max:  2.0

base_area:
  x_min: -2.0
  x_max:  0.0
  y_min:  0.0
  y_max:  2.0

topics:
  fire_state: /world/fire/state
  target_state: /world/target/state
  base_pose: /world/base/pose

robots:
  - id: Rescue_Limo_1
    role: rescue
    label: Rescue Limo 1
    short_label: R
    pose_topic: /Rescue_Limo_1/pose_world

  - id: Fire_Limo_1
    role: fire
    label: Fire Limo 1
    short_label: F1
    pose_topic: /Fire_Limo_1/pose_world

  - id: Fire_Limo_2
    role: fire
    label: Fire Limo 2
    short_label: F2
    pose_topic: /Fire_Limo_2/pose_world

visual:
  draw_video_overlay: true
  draw_map: true
  draw_connectors: true

  fire_suppress_duration_sec: 2.0

  video_robot_ring_radius_px: 26
  video_robot_ring_thickness_px: 3
  video_robot_ellipse_ratio: 0.55

  map_robot_length_px: 34
  map_robot_width_px: 22
  map_fire_radius_px: 18
  map_target_size_px: 18

  alpha_base: 0.28
  alpha_fire: 0.55
  alpha_connector: 0.35
```

---

## 15. CLI 예시

### Bag 정보 확인

```bash
python -m limo_ar.render --config configs/example_limo.yaml --inspect
```

출력:
- topic list
- duration
- detected robots
- target events
- fire events

### 짧은 preview

```bash
python -m limo_ar.render \
  --config configs/example_limo.yaml \
  --mode wide_with_map \
  --output outputs/preview_10s.mp4 \
  --start-sec 0 \
  --end-sec 10
```

### AR only

```bash
python -m limo_ar.render \
  --config configs/example_limo.yaml \
  --mode ar_only \
  --output outputs/ar_only.mp4
```

### Wide with map

```bash
python -m limo_ar.render \
  --config configs/example_limo.yaml \
  --mode wide_with_map \
  --output outputs/wide_with_map.mp4
```

---

## 16. Implementation notes

### 16.1 원본 영상 비율 유지

원본 영상을 정사각형으로 crop하거나 stretch하지 않는다.

- `ar_only`: 원본 video width/height 그대로 output
- `wide_with_map`: 원본 video width/height 그대로 왼쪽 배치, 오른쪽에 `height x height` map 추가

### 16.2 Drawing order

권장 drawing order:

1. video frame read
2. base overlay
3. target overlay
4. fire overlay
5. robot ring/ellipse
6. robot label
7. rescued target squares
8. map 생성
9. connector lines
10. final canvas write

Connector는 final canvas 단계에서 그린다.  
video 좌표와 map 좌표가 모두 final canvas 좌표로 변환된 뒤 그리는 것이 쉽다.

### 16.3 Audio

1차 구현에서는 audio 보존을 하지 않는다.  
필요하면 나중에 ffmpeg로 원본 audio를 mux한다.

### 16.4 Performance

1차는 정확성과 단순성을 우선한다.

- frame-by-frame OpenCV rendering
- tqdm progress bar 정도만 사용
- multiprocessing, GPU, caching은 하지 않는다

### 16.5 Missing homography behavior

`draw_video_overlay: true`인데 homography file이 없으면:

- hard fail 하지 말고 경고 출력
- video에는 robot/fire/target overlay를 생략하거나
- only label/status 없이 원본 영상 유지
- map은 정상 생성

다만 `ar_only` 모드에서 homography가 없으면 AR 요소가 거의 없으므로 사용자에게 명확한 경고를 출력한다.

---

## 17. Dependencies

권장 최소 dependency:

```text
opencv-python
numpy
pyyaml
tqdm
rosbags
```

가능하면 ROS 2 설치를 요구하지 않게 만든다.

---

## 18. Acceptance checklist

1차 구현 완료 기준:

- [ ] `.lrf` 또는 `.mp4`를 OpenCV로 읽을 수 있다.
- [ ] `.db3`에서 robot pose, fire state, target state를 읽을 수 있다.
- [ ] `ar_only` 모드로 원본 크기의 output video를 만들 수 있다.
- [ ] `wide_with_map` 모드로 `video_width + video_height` by `video_height` output video를 만들 수 있다.
- [ ] RViz-like map에 base, targets, fires, robots가 표시된다.
- [ ] fire가 inactive 되기 직전 2초 동안 shrinking animation으로 표시된다.
- [ ] target이 완료되면 사라진다.
- [ ] Rescue Limo 주변에 completed target count가 노란 square로 표시된다.
- [ ] Base 안에 들어온 robot 수에 따라 base 밝기가 변한다.
- [ ] video와 map의 같은 robot을 반투명 connector line으로 연결할 수 있다.
- [ ] `time_offset_sec`로 video-bag sync를 조정할 수 있다.
- [ ] config만 바꿔서 다른 video/bag 조합에 재사용할 수 있다.

---

## 19. 피해야 할 것

1차 구현에서는 다음을 하지 않는다.

- 웹 UI
- 실시간 ROS node
- 복잡한 dashboard
- 다중 camera support
- 3D rendering
- object detection 기반 robot tracking
- 딥러닝 모델
- 자동 homography 추정
- audio muxing
- 과도한 class hierarchy
- 과도한 플러그인 구조

목표는 **작고 명확한 offline renderer**다.
