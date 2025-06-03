import json

json_path = 'data1.json'

# JSON 파일 불러오기
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

    print(f"Total number of data entries: {len(data)}")

    # frame 값이 연속적인지 확인하고, 누락된 frame에 대해 이전 데이터를 복사하여 보간
    frames = [item['frame'] for item in data]
    min_frame = min(frames)
    max_frame = max(frames)

    frame_dict = {item['frame']: item for item in data}
    interpolated_data = []

    for i in range(min_frame, max_frame + 1):
        if i in frame_dict:
            interpolated_data.append(frame_dict[i])
            prev_item = frame_dict[i]
        else:
            # 누락된 frame이면 이전 데이터를 복사해서 frame만 변경
            new_item = prev_item.copy()
            new_item['frame'] = i
            interpolated_data.append(new_item)

    # 결과를 새로운 파일로 저장
    output_path = 'data1.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(interpolated_data, f, ensure_ascii=False, indent=2)

    print(f"Number of entries after interpolation: {len(interpolated_data)}")