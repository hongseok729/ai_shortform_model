import cv2
import os
import time

def count_frames_manual(video_path):
    """실제 비디오의 프레임을 모두 읽으면서 카운트"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return 0
    
    frame_count = 0
    start_time = time.time()
    
    # 모든 프레임 순회
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # 진행 상황 표시 (100프레임마다)
        if frame_count % 100 == 0:
            print(f"처리 중: {frame_count}개 프레임 읽음...")
    
    cap.release()
    
    end_time = time.time()
    print(f"카운팅 시간: {end_time - start_time:.2f}초")
    
    return frame_count

def save_sample_frames(video_path, output_dir, interval=100):
    """일정 간격으로 프레임 샘플 저장"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 일정 간격으로 프레임 저장
        if frame_count % interval == 0:
            output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            print(f"프레임 저장: {output_path}")
        
        frame_count += 1
    
    cap.release()
    print(f"총 {saved_count}개의 샘플 프레임 저장 완료")

def main():
    video_path = "../input_data/data_1.mp4"
    output_dir = "../frame_samples2"
    
    # 비디오 정보 출력
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return
    
    # 비디오 메타데이터 확인
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("===== 비디오 정보 =====")
    print(f"해상도: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"OpenCV 보고 프레임 수: {reported_frames}")
    print(f"예상 길이: {reported_frames/fps:.2f}초")
    
    cap.release()
    
    # 실제 프레임 카운트
    print("\n===== 실제 프레임 카운팅 시작 =====")
    actual_frames = count_frames_manual(video_path)
    print(f"실제 카운팅된 프레임 수: {actual_frames}")
    print(f"실제 길이: {actual_frames/fps:.2f}초")
    
    # 차이 확인
    if reported_frames != actual_frames:
        print(f"\n주의: 프레임 수 불일치!")
        print(f"차이: {abs(reported_frames - actual_frames)}개")
        print(f"비율: {actual_frames/reported_frames*100:.2f}%")
    
    # 샘플 프레임 저장 (선택적)
    save_frames = input("\n샘플 프레임을 저장하시겠습니까? (y/n): ")
    if save_frames.lower() == 'y':
        interval = int(input("몇 프레임 간격으로 저장할까요? (기본값: 100): ") or 100)
        save_sample_frames(video_path, output_dir, interval)

if __name__ == "__main__":
    main()