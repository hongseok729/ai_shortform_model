import torch

def first_order_diff(frame, prev_frame):
    '''
    1차 차분 계산을 위해
    '''
    return torch.abs(frame - prev_frame)

def second_order_diff(frame, prev_frame2):
    '''
    2차 차분 계산을 위해
    '''
    return torch.abs(frame - prev_frame2)


#사실 시간만 된다면 n차분을 더 해서 테스트 해볼려고 했지만, 시간이 없습니다.