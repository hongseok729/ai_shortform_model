# import gradio as gr
# import subprocess
# import shutil
# import os

# def run_pipeline(video_file):
#     input_path = "../input/input.mp4"
#     os.makedirs("../input", exist_ok=True)

#     # 안전하게 복사
#     with open(input_path, "wb") as f:
#         f.write(video_file.read())
#     # 각 단계 실행
#     # subprocess.run(["python", "../step1_Shot_Boundary_Detection/inference.py"], check=True)
#     # subprocess.run(["python", "../step2_Face_Detection/MTCNN-face_detection.py"], check=True)
#     # subprocess.run(["python", "../step3_Main_Character/main3.py"], check=True)
#     # subprocess.run(["python", "../step4_Face_Tracking/main4.py"], check=True)
#     # subprocess.run(["python", "../step5_Stable_Croping/main4.py"], check=True)

#     # 출력 영상 경로
#     output_path = "../output/output_vertical.mp4"
#     return output_path if os.path.exists(output_path) else None

# with gr.Blocks(title="AI 세로 영상 편집기") as demo:
#     with gr.Row():
#         with gr.Column(scale=1):
#             video_input = gr.Video(label="영상 파일 업로드", interactive=True)
#             run_button = gr.Button("AI 편집 실행")

#         with gr.Column(scale=2):
#             video_preview = gr.Video(label="세로 편집 결과", interactive=False)

#     with gr.Row():
#         timeline = gr.Slider(label="타임라인", minimum=0, maximum=100, step=1)
#         cut_button = gr.Button("컷 편집")
#         export_button = gr.Button("결과 저장")

#     run_button.click(fn=run_pipeline, inputs=video_input, outputs=video_preview)

# demo.launch()


import os
import shutil
import subprocess
import sys

def run_pipeline(input_video_path):
    input_dir = "./input"
    output_path = "./output/output_vertical.mp4"
    os.makedirs(input_dir, exist_ok=True)

    # 입력 영상 복사
    input_copy_path = os.path.join(input_dir, "input.mp4")
    shutil.copy(input_video_path, input_copy_path)
    print(f"\n[INFO] 입력 영상 복사 완료: {input_copy_path}")

    # 단계별 실행 명령어
    steps = [
        ("Shot Boundary Detection", ["python", "step1_Shot_Boundary_Detection/inference.py"]),
        ("Face Detection", ["python", "step2_Face_Detection/MTCNN-face_detection.py"]),
        ("Main Character Selection", ["python", "step3_Main_Character/main3.py"]),
        ("Face Tracking", ["python", "step4_Face_Tracking/main4.py"]),
        ("Stable Cropping", ["python", "step5_Stable_Croping/main4.py"])
    ]

    for i, (desc, cmd) in enumerate(steps, 1):
        input(f"\n[STEP {i}] '{desc}' 단계입니다. 계속하려면 Enter를 누르세요...")
        print(f"[실행 중] {desc} → {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[완료] {desc}")
        except subprocess.CalledProcessError as e:
            print(f"[오류] {desc} 실패: {e}")
            sys.exit(1)

    # 결과 확인
    if os.path.exists(output_path):
        print(f"\n[완료] 최종 출력 파일 경로: {output_path}")
    else:
        print("\n[오류] 출력 파일이 생성되지 않았습니다.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python pipeline_interactive.py <영상파일경로>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    if not os.path.exists(input_video_path):
        print(f"[오류] 파일이 존재하지 않음: {input_video_path}")
        sys.exit(1)

    run_pipeline(input_video_path)
