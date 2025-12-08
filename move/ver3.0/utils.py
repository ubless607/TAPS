def pwm_to_degree(pwm_value):
    """
    다이나믹셀 PWM(0~4095)을 각도(Degree)로 변환
    XL430/330 기준: 1 tick = 0.088도
    """
    # 0 ~ 4095 -> 0 ~ 360도
    return pwm_value * 0.088