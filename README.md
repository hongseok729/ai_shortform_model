## step4_Face Tracking > best_orbit_tracking.py

# 목적
이 프로젝트는 영상 속 주인공의 얼굴을 안정적으로 추적하기 위한 알고리즘입니다.
AI 숏폼 콘텐츠 자동 생성 시스템에서, "누가 주인공인가"를 판단한 후 그 주인공을 프레임 단위로 안정적으로 따라가는 tracking 로직이 핵심이며, 본 코드는 그 문제를 해결하기 위해 설계되었습니다.

# 알고리즘 핵심 개요
이 Face Tracking 시스템은 다음 네 가지 핵심 전략을 결합하여 동작합니다:

1. 궤도 기반 보간 (Orbit-Based Interpolation)
각 샷(shot)마다 주인공 얼굴의 중심 좌표를 모아서, 프레임 중심을 이차 보간(quadratic interpolation) 으로 예측합니다.

이로 인해 주인공의 얼굴이 명시적으로 존재하지 않는 프레임에서도 예상 위치를 추정할 수 있습니다.

2. Fallback 복원 전략
주인공 정보가 없는 프레임에서, 주변에 검출된 얼굴들을 평가합니다.

# 평가 방식

중심과의 거리

face 영역 크기 유사도

detection confidence

이 모든 요소를 조합한 스코어 기반 평가로 가장 유력한 후보를 선택합니다.

3. 중심 예측 + 크기 제어
이전 박스(prev_box)가 존재하면, 중심이 급격히 이동하지 않도록 제한된 범위(CENTER_DRIFT_LIMIT, Y축 별도 제한) 내에서 위치를 이동시킵니다.

bbox의 크기도 초과하면 제한하여 너무 커지거나 작아지는 문제를 방지합니다.

4. 부드러운 이동 (스무딩)
프레임 간 bbox가 급격히 튀는 현상을 막기 위해, 이전 박스와 현재 박스를 보간합니다.

보간 계수 SMOOTHING_ALPHA를 통해 매끄럽고 자연스러운 크롭 영상이 가능하게 됩니다.

# 입출력 데이터
# 입력:
main_characters1.json: 각 샷별로 주인공이 명시된 프레임 정보 ("shot", "frame", "main": [x1, y1, x2, y2])

faces_detection1.json: 매 프레임마다 검출된 전체 얼굴들의 좌표와 confidence

# 출력:
data1.json: 프레임 단위로 주인공 얼굴 bounding box 결과가 기록됨

json
[
  {
    "frame": 135,
    "shot": 4,
    "main": [x1, y1, x2, y2]
  },
  ...
]

이 결과를 활용하여 영상에서 주인공 얼굴을 중심으로 crop한 숏폼을 생성할 수 있습니다.
