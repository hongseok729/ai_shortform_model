# 각 프레임을 개별적으로 처리
    if mode == 1:
        # 비디오 출력 스트림 설정 (크롭된 크기)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (output_width, output_height))
        
        processed_frames = 0
        
        for frame_idx, frame in all_frames:
            try:
                # 현재 프레임에 해당하는 크롭 영역 찾기
                crop_info = next((region for region in crop_regions if region['frame_idx'] == frame_idx), None)
                
                if crop_info:
                    # 크롭 영역 정보
                    x, y = crop_info['x'], crop_info['y']
                    crop_width, crop_height = crop_info['width'], crop_info['height']
                    
                    # 홀수 너비 처리
                    if crop_width % 2 != 0:
                        crop_width += 1
                    
                    # 안전한 크롭 함수 사용
                    cropped_frame = safe_crop(frame, x, y, crop_width, crop_height, output_width, output_height)
                    
                    # 프레임 번호 표시
                    cropped_frame = draw_frame_number(cropped_frame, frame_idx)
                    
                    # 출력 비디오에 쓰기
                    out.write(cropped_frame)
                    processed_frames += 1
                else:
                    # 크롭 정보가 없는 프레임은 중앙 크롭 적용
                    center_x = frame.shape[1] // 2
                    x = max(0, min(center_x - output_width // 2, frame.shape[1] - output_width))
                    
                    # 안전한 크롭 함수 사용
                    cropped_frame = safe_crop(frame, x, 0, output_width, output_height, output_width, output_height)
                    
                    # 프레임 번호 표시
                    cropped_frame = draw_frame_number(cropped_frame, frame_idx)
                    
                    # 출력 비디오에 쓰기
                    out.write(cropped_frame)
                    processed_frames += 1
            except Exception as e:
                print(f"프레임 {frame_idx} 처리 중 오류 발생: {e}")
                # 검은색 프레임 생성
                black_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                black_frame = draw_frame_number(black_frame, frame_idx)
                out.write(black_frame)
                processed_frames += 1
            
            # 진행 상황 표시
            if frame_idx % 100 == 0 or frame_idx == frame_count - 1:
                print(f"처리 중: {frame_idx+1}/{frame_count} 프레임")
        
        # 자원 해제
        out.release()
    else:  # 모드 2
        # 비디오 출력 스트림 설정 (원본 크기 유지)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        processed_frames = 0
        
        for frame_idx, frame in all_frames:
            try:
                # 현재 프레임에 해당하는 크롭 영역 찾기
                crop_info = next((region for region in crop_regions if region['frame_idx'] == frame_idx), None)
                
                if crop_info:
                    # 크롭 영역 정보
                    x, y = crop_info['x'], crop_info['y']
                    crop_width, crop_height = crop_info['width'], crop_info['height']
                    
                    # 홀수 너비 처리
                    if crop_width % 2 != 0:
                        crop_width += 1
                    
                    # 경계 확인 및 조정
                    if x < 0 or y < 0 or x + crop_width > frame.shape[1] or y + crop_height > frame.shape[0]:
                        # 경계를 벗어나는 경우 조정
                        x = max(0, min(x, frame.shape[1] - crop_width))
                        y = max(0, min(y, frame.shape[0] - crop_height))
                        
                        # 추가 검증: 크기가 0이 아닌지 확인
                        if crop_width <= 0 or crop_height <= 0:
                            # 안전한 기본값 사용
                            crop_width = min(frame.shape[1], output_width)
                            crop_height = min(frame.shape[0], output_height)
                    
                    # 프레임 번호 표시
                    frame = draw_frame_number(frame, frame_idx)
                    
                    # 크롭 영역을 Bounding Box로 표시
                    frame = draw_crop_box(frame, x, y, crop_width, crop_height)
                    
                    # 현재 프레임에 해당하는 얼굴 트래킹 정보 찾기
                    if frame_idx in tracking_dict:
                        face_info = tracking_dict[frame_idx]
                        face_x = face_info['x']
                        face_y = face_info['y']
                        face_width = face_info['width']
                        face_height = face_info['height']
                        
                        # 얼굴 영역을 Bounding Box로 표시
                        frame = draw_face_box(frame, face_x, face_y, face_width, face_height)
                    
                    # 출력 비디오에 쓰기
                    out.write(frame)
                    processed_frames += 1
                else:
                    # 크롭 정보가 없는 프레임은 중앙에 가상의 크롭 영역 표시
                    center_x = frame.shape[1] // 2
                    x = max(0, min(center_x - output_width // 2, frame.shape[1] - output_width))
                    y = 0 # 세로는 항상 상단부터 시작
                    
                    # 프레임 번호 표시
                    frame = draw_frame_number(frame, frame_idx)
                    
                    # 가상의 크롭 영역을 Bounding Box로 표시
                    frame = draw_crop_box(frame, x, y, output_width, output_height)
                    
                    # 현재 프레임에 해당하는 얼굴 트래킹 정보 찾기
                    if frame_idx in tracking_dict:
                        face_info = tracking_dict[frame_idx]
                        face_x = face_info['x']
                        face_y = face_info['y']
                        face_width = face_info['width']
                        face_height = face_info['height']
                        
                        # 얼굴 영역을 Bounding Box로 표시
                        frame = draw_face_box(frame, face_x, face_y, face_width, face_height)
                    
                    # 출력 비디오에 쓰기
                    out.write(frame)
                    processed_frames += 1
            except Exception as e:
                print(f"프레임 {frame_idx} 처리 중 오류 발생: {e}")
                # 검은색 프레임 생성
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                black_frame = draw_frame_number(black_frame, frame_idx)
                out.write(black_frame)
                processed_frames += 1
            
            # 진행 상황 표시
            if frame_idx % 100 == 0 or frame_idx == frame_count - 1:
                print(f"처리 중: {frame_idx+1}/{frame_count} 프레임")
        
        # 자원 해제
        out.release()