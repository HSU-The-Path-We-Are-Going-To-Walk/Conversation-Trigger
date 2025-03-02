import cv2
import torch

# YOLOv5 모델 불러오기 (Ultralytics에서 제공하는 pretrained 모델 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 웹캠 또는 비디오 스트림 열기 (0은 기본 웹캠)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("비디오 스트림을 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV는 BGR 포맷이므로, RGB로 변환
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv5 모델로 실시간 추론 실행
    results = model(img_rgb)

    # 결과를 pandas DataFrame으로 변환 (각 행은 감지된 객체)
    df = results.pandas().xyxy[0]

    # 'person'으로 필터링 (COCO 데이터셋 기준)
    persons = df[df['name'] == 'person']

    # 사람이 정확히 1명일 때
    if len(persons) == 1:
        cv2.putText(frame, "사람 1명 감지", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # 감지된 사람의 bounding box 그리기
        for _, row in persons.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"감지된 사람 수: {len(persons)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 결과 프레임 출력
    cv2.imshow('실시간 사람 탐지', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
