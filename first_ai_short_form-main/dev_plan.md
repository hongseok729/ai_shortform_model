# 필요 라이브러리 
pip install opencv-python mtcnn numpy scikit-image

# Day1
<details>
<summary>접기/펼치기</summary>
    
## Total

### 1) Shot Boundary Detection 모델 적용 (서영현)
- 원본 mp4 파일에서 shot/scene 전환 구간 추출
- **Input**
    - 1분 미만 분량의 .mp4 파일 (가로 영상)
- **Output**  
    `[ddhhmmss:ddhhmmss, ddhhmmss:ddhhmmss, …]`  
    (예: `[000012:000025, 000026:000040, ...]`)
- **Result**  
    - 원본 mp4 파일
    - shot/scene 전환 구간 리스트

### 2) Face Detection 모델 적용 (이예림)
- 프레임별 얼굴 검출 및 바운딩 박스 좌표 출력
- **Input**
    - 원본 mp4 파일
    - shot/scene 전환 구간 리스트
- **Output**  
    - 원본 mp4 파일
    - shot/scene 전환 구간 리스트
    - 바운딩 박스 리스트
    - 예) `[
  {"frame": 12, "faces": [[x1, y1, x2, y2, conf], [x1, y1, x2, y2, conf], ...]},
  {"frame": 13, "faces": [[x1, y1, x2, y2, conf], ...]},
  ...
]
- **Result**  
    - 원본 mp4 파일
    - shot/scene 전환 구간 리스트
    - 각 프레임별 face bounding box 좌표 정보

### 3) 주인공 선정 알고리즘 개발 (이예림)
- 원본 mp4 및 face detection 결과 기반
- 등장 인물 중 **주인공(main)** 지정
    - 주인공 결정
- **Input**
    - 원본 mp4 파일 
    - shot/scene 전환 구간 리스트
    - 바운딩 박스 리스트
- **Output**  
    - 원본 mp4 파일
    - shot, scene 전환 구간 리스트
    - 주인공 얼굴에 바운딩박스가 그려진 이미지 파일
    - 예) `frame_0012_main.jpg`
    - 주인공 얼굴 이미지의 바운딩박스 좌표
    - 예) `[
  {"frame": 12, "main": [320, 120, 420, 240]}
]`
- **Result**  
    - 원본 mp4 파일
    - shot, scene 전환 구간 리스트
    - 주인공 얼굴에 바운딩박스가 그려진 이미지 파일
    - 주인공 얼굴 이미지의 바운딩박스 좌표

### 4) Face Tracking 알고리즘 개발 (전홍석)
- shot/scene별로 main 인물이 유지되도록 tracking
- **Input**
    - 원본 mp4 파일
    - shot/scene 전환 구간 리스트
    - 주인공 얼굴 이미지의 바운딩박스 좌표
    - 예) `[
  {"frame": 12, "main": [320, 120, 420, 240]}
]`
- **Output**
    - 각 프레임 별 주인공 얼굴의 바운딩 박스 좌표 리스트
    - 예) '[{"frame": 12, "main": [320, 120, 420, 240]}, {"frame": 13, "main": [325, 122, 425, 242]}, ...])'
    - 주인공 얼굴이 추적된 결과 영상(mp4)
        - 프레임마다 주인공 얼굴에 바운딩박스가 표시된 mp4영상 파일
    - (고려중) 주인공 tracking confidence점수 또는 누락 프레임 로그
- **Result**  
    - 원본 mp4 파일
    - shot/scene 전환 구간 리스트
    - 주인공 얼굴이 프레임마다 tracking된 좌표 정보
    - 주인공 tracking 결과가 적용된 출력 영상 파일

### 5) Stable Cropping 알고리즘 개발 (정재원)
- 가로 영상을 **세로 영상(숏폼)**으로 변환 ( 9:16 )
- main 인물을 중심으로 안정적으로 crop
- 숏폼의 세로 길이를 원본 가로 영상의 세로 길이 그대로?
- **Input**  
    - main 인물의 frame별 face bounding box (center_x, w 필수)  
- **Output**  
- **Result**
    - 숏폼 형식으로 새로 재구성된 영상 mp4 file  

### 추가 개발 (부가 요소)
- LLM(대형언어모델) 활용
- STT(음성 인식)
- 화질 보정
- 자막 자동 생성 등

## Individual

### 서영현
- 금일 작업 내용: readme 작성 input, output 등 포맷 결정, 자료 수집, 알고리즘 분석
- 어려웠던 점: 논문 알고리즘 분석 후 최적의 포맷 결정, 양질의 자료 분류
- 내일 작업 계획: 분석한 논문을 코드로 구현 

### 이예림
- 금일 작업 내용 : 프로젝트 분석 및 역할 분배
- 어려웠던 점 : 프로젝트 전체적 흐름 이해 및 기술 분석
- 내일 작업 계획 : 주인공 선정 알고리즘 논문 분석 및 개발 시작

### 정재원
- 금일 작업 내용 : Stable Cropping 알고리즘 기본 구조 구상
- 어려웠던 점 : Critical Value 발생시 ANIMATION_MOVE 관련 부분 구상
- 내일 작업 계획 : Cropping 관련 논문 탐색 완료 후 소스 코드 구현 시작

### 전홍석
- 금일 작업 내용 : Face Tracking 알고리즘 분석 (관련 알고리즘/레퍼런스 분석 진행) , 관련 논문 참조 
- 어려웠던 점 : shot 전환 시 tracking ID가 바뀌는 구조적 문제 발생 가능성 예상
- 내일 작업 계획 : 추적 후보 알고리즘 비교해보기
SORT, Deep SORT, ByteTrack, Custom tracker (feature matching)
</details>



# Day2
<details>
<summary>접기/펼치기</summary>
  
## Daily Standup

    
## Total

### 1) Shot Boundary Detection 모델 적용 (서영현)
- 원본 mp4 파일에서 shot/scene 전환 구간 추출
- **Input**
    - 1분 미만 분량의 .mp4 파일 (가로 영상)
- **Output**  
    `[
  {"start": "00:00:00.00", "end": "00:00:05.00", "shot": 0},
  {"start": "00:00:05.00", "end": "00:00:08.00", "shot": 1},
  {"start": "00:00:08.00", "end": "00:00:25.00", "shot": 2},
  ...
]`  
    (예: `[
  {"start": "시:분:초.밀리초", "end": "시:분:초.밀리초", "shot": 0}, ...]` )
- **Result**  
    - 원본 mp4 파일
    - shot/scene 전환 구간 json

### 2) Face Detection 모델 적용 (이예림)
- 프레임별 얼굴 검출 및 바운딩 박스 좌표 출력
- **Input**
    - 원본 mp4 파일
    - shot/scene 전환 구간 리스트
- **Output**  
    - 원본 mp4 파일
    - shot/scene 전환 구간 json
    - 바운딩 박스 json
    - 예) `[
  {"shot":0, "frame": 12, "faces": [[x1, y1, x2, y2, conf], [x1, y1, x2, y2, conf], ...]},
  {"shot":0, "frame": 13, "faces": [[x1, y1, x2, y2, conf], ...]},
  ...
]` 
- **Result**  
    - 원본 mp4 파일
    - shot/scene 전환 구간 리스트
    - 각 프레임별 face bounding box 좌표 정보

### 3) 주인공 선정 알고리즘 개발 (이예림)
- 원본 mp4 및 face detection 결과 기반
- 등장 인물 중 **주인공(main)** 지정
    - 주인공 결정
- **Input**
    - 원본 mp4 파일 
    - shot/scene 전환 구간 리스트
    - 바운딩 박스 리스트
- **Output**  
    - 원본 mp4 파일
    - shot, scene 전환 구간 리스트
    - 주인공 얼굴에 바운딩박스가 그려진 이미지 파일
    - 예) `shot00_frame_0012_main.jpg`
    - 주인공 얼굴 이미지의 바운딩박스 좌표
    - 예) `[
  {"shot":0, "frame": 12, "main": [x1, y1, x2, y2, conf]},
  {"shot":1, "frame": 13, "main": [x1, y1, x2, y2, conf]},
  {"shot":2, "frame": 13, "main": [x1, y1, x2, y2, conf]}, ...
]`
- **Result**  
    - 원본 mp4 파일
    - shot, scene 전환 구간 json
    - shot 별 주인공 얼굴에 바운딩박스가 그려진 이미지 파일
    - shot 별 주인공 얼굴 이미지의 바운딩박스 좌표
    - 전체 face_detection.json

### 4) Face Tracking 알고리즘 개발 (전홍석)
- shot/scene별로 main 인물이 유지되도록 tracking
- **Input**
    - 원본 mp4 파일
    - shot/scene 전환 구간 리스트
    - 주인공 얼굴 이미지의 바운딩박스 좌표
    - 예) `[
  {"frame": 12, "main": [320, 120, 420, 240]}
]`
- **Output**
    - 각 프레임 별 주인공 얼굴의 바운딩 박스 좌표 리스트
    - 예) '[{"frame": 12, "main": [320, 120, 420, 240]}, {"frame": 13, "main": [325, 122, 425, 242]}, ...])'
    - 주인공 얼굴이 추적된 결과 영상(mp4)
        - 프레임마다 주인공 얼굴에 바운딩박스가 표시된 mp4영상 파일
- **Result**  
    - 원본 mp4 파일
    - shot/scene 전환 구간 리스트
    - 주인공 얼굴이 프레임마다 tracking된 좌표 정보
    - 주인공 tracking 결과가 적용된 출력 영상 파일

    **( 샷 별로 ) 주인공을 다르게 설정하여 tracking 알고리즘 구현 중**
### 5) Stable Cropping 알고리즘 개발 (정재원)
- 가로 영상을 **세로 영상(숏폼)**으로 변환 ( 9:16 )
- 특정 시점 (Shot의 변화 or Target Face가 바뀌는 시점)에 Target Face를 추적, 해당 Face BBox의 Center를 중심으로 잡는 Crop box 최초 지정
- 이후 다음 시점이 지정되기 전까지 Target Face(의 중심점)의 이동을 추적하여 Crop box 조정  
- 숏폼의 세로 길이를 원본 가로 영상의 세로 길이 그대로!
- **Input**  
    - main 인물의 frame별 face bounding box (center_x, w 필수)  
    - Target Face BBox를 바꿀 시점 (Shot 변화가 이루어지는 Frame의 정보)  
- **Output**
    - 매 Frame별 Crob box의 x, y, w, h가 저장된 값 -> Result의 영상 자르기에 활용
- **Result**
    - 숏폼 형식으로 새로 재구성된 영상 mp4 file  

### 추가 개발 (부가 요소)
- LLM(대형언어모델) 활용
- STT(음성 인식)
- 화질 보정
- 자막 자동 생성 등

## Individual

### 서영현

### 이예림
[Day2] Face Detectin 모델 이용 개발
* 금일 작업 내용 요약 : 
- yolov8n , MTCNN 테스트 후 속도, 정확도 측면에서 yolov8n 선정 
- yolo 디폴트 바운딩 박스는 얼굴 영역이 너무 적어 확장 
- 바운딩박스의 좌표 안정을 위해 임계값 조정
- step1 , step3 과 원활한 데이터 교환을 위해 input output 정리
- 프레임 별 얼굴 바운딩 박스에서 main 선정 알고리즘 개발
- 바운딩 박스 크기, 바운딩박스의 중앙화 바운딩박스 얼굴의 빈도수 계산 및 가중치 부여 후 main 얼굴 도출

* 해결한 문제 :
* face detecting 모델 선정 완료
* step 1, 2, 3 input-output 정리 완료

* 어려웠던 점: 
- yolo를 사용한 얼굴 인식 바운딩 박스가 뜬금없이 튀거나, 엉뚱한 곳을 잡는다. 해당 예외처리가 아직 해결되지 않음
- 여러 인물이 나오고 얼굴 크기가 일정할 때 main 얼굴 도출 알고리즘이 정상적으로 작동하지 않음..
- 비슷한 scene에서 메인 얼굴이 바뀌는 경우가 있음
- 화면 전반적으로 크게 잡힌 얼굴이나, 가중치에 따라 메인으로 잡히지 않는 경우가 있음

* 내일 작업 계획
- yolo 바운딩 박스 시 튀는 값, 얼굴이 아닌 값 예외처리 로직 처리
- 중앙에 가깝고 화면에 크게 잡힌 얼굴이 있을 때 다른 얼굴 바운딩박스에 대해 main이라고 인식하는 결과가 나오지 않도록 알고리즘 수정
- step 1, 2, 3, 4 연계 후 테스트



### 정재원
- 금일 작업 내용 요약 : Stable Cropping 부분 코드 초안 구현.  
- 특정 시점 주인공의 Face bbox를 탐지후 crop box 설정 후 frame별 조정.  
- 어려웠던 점 : Crop box가 영상 범위를 벗어날 가능성 등 예외 처리가 많음. 또한 지속적인 예외 사항 발굴 필요.  
- 내일 작업 계획 : Crop Box의 부드러운 움직임을 위한 Threshold 및 speed값 조정 및 추가 데이터 받은 후 지속 검증 필요. 

### 전홍석
금일 작업 내용 : 
논문 Face Tracking -> DeepSORT basic 알고리즘 생성, 한계점에 관한 강화 알고리즘 생성 
(한계점에 관한 강화 알고리즘 logic)
1. Appearance 유사 객체 간 ID 혼동 -> YOLO(?) 탐지된 얼굴들 중 주인공 후보와 cosine similarity 비교 ( 유사도 높은 face만 주인공으로 추적)
2. 외형 변화에 약하다. (정적 임베딩) -> EMA 방식 embedding 업데이트 적용 (main_embedding = 0.95 * old + 0.05 * new)
3. 가림(occlusion)시 ID 스위치 -> DeepSORT의 Kalman Filter + cosine similarity 로 re-identification 가능
4. 감지 실패 시 대처 없음 -> 일정 similarity 이하인 경우 tracking에서 제외 or 로그 누락 처리
5. 파라미터 튜닝 민감성 -> similarity threshold 0.85로 설정 (안정성 확보)
어려웠던 점 : shot 전환시 새로운 track_id 생성 ( re-identification 로직으로 인해 )
내일 작업 계획 : 샷 별로 주인공을 다르게 설정하여 tracking 알고리즘 구현 
</details>



# Day3
<details>
<summary>접기/펼치기</summary>
  
## Daily Standup

    
## Total

### 1) Shot Boundary Detection 모델 적용 (서영현)

### 2) Face Detection 모델 적용 (이예림)

### 3) 주인공 선정 알고리즘 개발 (이예림)

### 4) Face Tracking 알고리즘 개발 (전홍석)

### 5) Stable Cropping 알고리즘 개발 (정재원)

## Individual

### 서영현

### 이예림
### 정재원
### 전홍석

### 4)

1. 주인공 궤도 예측 (Orbit Interpolation)
샷(shot)별로 주어진 주인공의 바운딩 박스들에서 중심좌표(cx, cy)를 뽑아낸 뒤, 3개 이상의 좌표가 존재하는 경우 이를 기반으로 2차 보간(quadratic interpolation) 을 수행한다. 이렇게 만든 보간 곡선을 이용해 해당 샷의 모든 프레임에 대해 주인공이 등장할 것으로 예측되는 위치를 계산한다.
→ 주인공 bbox가 명확하게 주어지지 않은 프레임에서도 일관된 위치 예측이 가능해진다.

2. 프레임별 주인공 후보 선택 및 fallback 처리
각 프레임마다 다음 순서로 주인공 bbox를 결정한다:

(1) main_characters1.json에 프레임이 존재하면 해당 bbox 사용.

(2) 얼굴이 감지된 경우 faces_detection1.json에서 confidence가 높은 얼굴들을 필터링한 뒤, 예측 중심좌표(pred_cx, pred_cy)와의 거리, bbox 면적 유사도를 고려해 가장 적합한 bbox를 선택한다.

(3) 아무 얼굴도 감지되지 않은 경우에는 이전 bbox 위치를 기준으로, 보간된 궤도를 따라 **예측 중심에 bbox를 생성(fallback)**한다. 이 fallback은 최대 10프레임까지만 허용하며, 점점 크기를 줄이며 bbox가 튀지 않도록 한다.

3. 부드러운 bbox 전환 및 JSON 저장
예측된 bbox와 이전 프레임의 bbox가 동시에 존재하면, 두 박스를 α=0.8의 비율로 혼합해 **부드러운 이동(Smoothing)**을 적용한다. 또한, 예측된 중심과 bbox 중심이 너무 멀어지면 bbox를 강제로 궤도 예측 위치로 재정렬해 급격한 튐 현상을 방지한다.
최종적으로 각 프레임에 대해 결정된 주인공 bbox를 frame, shot, main으로 구성된 딕셔너리로 저장하고, 전체를 JSON 파일로 출력한다.

</details>


# Day4
<details>
<summary>접기/펼치기</summary>
  
## Daily Standup
새벽간 변동사항을 확인하였으며 특이사항은 따로 존재하지 않았고 금일 각자가 진행할 내용및 본인이 참고한 논문등 레퍼런스에대한 이야기를 진행함
    
## Total

### 1) Shot Boundary Detection 모델 적용 (서영현)
다양한 영상에도 동작하도록 각종 수치 수정을 진행하였습니다.
1. 이전프레임과 비교하여 일정 수치 이상 변동시 화면 전환으로 인식
2. 프레임의 pre 값이 일정수치 이상일시 화면 전환으로 인식
3. pre값이 일정수치 이하일시 이전수치와의 변동값이 일정수치 이상이여도 화면전환 아님
등을 진행하였습니다.

### 2) Face Detection 모델 적용 (이예림)

### 3) 주인공 선정 알고리즘 개발 (이예림)

### 4) Face Tracking 알고리즘 개발 (전홍석)

### 5) Stable Cropping 알고리즘 개발 (정재원)

## Individual

### 서영현
금일 작업 요약
- output 수정
- 학습 속도를 위하여 기존 영상학습에서 이미지 학습을 위한 코드 추가
- 다양한 영상에서 동작하도록 수치 수정
- 레이어 최종 확인 완료
- 탈출 조건 확인 및 탈출 확인
- 학습 진행 상황 확인

어려웠던 점
- 다양한 영상에 대응하도록 고도화 하는게 어려웠다
- 논문을 확인한 후 레이어 일치 여부 확인이 어려웠다
- 실제 구현시 결과가 나오기까지 맞는 로직을 구현했는지 확인하는게 어려웠다

내일 작업 계획
- 파이프라인 구성
- 아직 미완성된 다른 코드들 구현
- 
### 이예림
### 정재원
### 전홍석
</details>


