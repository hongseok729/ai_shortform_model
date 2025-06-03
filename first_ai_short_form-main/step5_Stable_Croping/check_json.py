import json
import os

json_path = 'data1.json'

if not os.path.exists(json_path):
    print(f"파일이 존재하지 않습니다: {json_path}")
else:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"데이터 타입: {type(data)}")
    if isinstance(data, list):
        print(f"총 데이터 수: {len(data)}")
    elif isinstance(data, dict):
        print(f"총 키 수: {len(data.keys())}")
    else:
        print("알 수 없는 데이터 구조입니다.")
