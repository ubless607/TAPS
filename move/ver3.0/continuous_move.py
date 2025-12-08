import sys
import select
import time
import numpy as np
import mujoco
import mujoco.viewer
import os
import math 

from interface import SimulatedRobot
from robot import Robot

port = 'COM3'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'low_cost_robot/scene.xml')
EE_SITE_NAME = 'joint6'

GRIPPER_OPEN_PWM  = 2979
GRIPPER_CLOSE_PWM = 1901
JOINT6_POSITION = None
# --- [설정 1] 오프셋 (User Settings) ---
JOINT_OFFSETS = np.array([0.00, 0.25, -1.57, 0.00, 2.29, 0.00])

# --- [설정 2] 모터 방향 ---
DIR = np.array([1, 1, 1, 1, 1, 1])

# --- [설정 3] 관절별 허용 범위 (User Settings) ---
JOINT_LIMITS = [
    (900, 3000),    # J1: 몸통
    (800, 2000),    # J2: 허리
    (1000, 3500),   # J3: 팔꿈치
    (700, 3500),    # J4: 손목 끄덕
    (0, 4095),      # J5: 손목 회전
    (1500, 3500)    # J6: 그리퍼
]

# --- [설정 4] 바닥 충돌 제한 높이 ---
FLOOR_LIMIT = 0.07

# --- [설정 5] 자동 탐색 설정 ---
PINCER_CONFIG = {
    'SWEEP_SPEED': 6,
    'COLLISION_THRESH': 60,
    'RIGHT_LIMIT_J0': 993,
    'LEFT_LIMIT_J0': 3000,
    'INITIAL_POSE': np.array([2048, 1623, 1171, 1289, 3000, 2130])
}

# --- [설정 6] 높이(Z) 탐색 설정 ---
Z_SEARCH_CONFIG = {
    'DESCEND_SPEED': 2,        # 하강 속도
    'COLLISION_THRESH': 60,    # 충돌 감지 민감도
    'LOWEST_J2_PWM': 850,      # 하강 한계
    'LIFT_J2_PWM': 1900        # 복귀 시 허리 높이
}

m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)
sim_robot = SimulatedRobot(m, d)


robot = Robot(device_name=port)

if os.name == 'nt':
    import msvcrt

robot._disable_torque()

# 초기화
pwm = np.array(robot.read_position())
d.qpos[:6] = sim_robot._pwm2pos(pwm) + JOINT_OFFSETS
mujoco.mj_forward(m, d)

teleop_active = True
curr_pwm = pwm.copy()
target_pwm = pwm.copy()

JOINT_SPEED = 6

def get_command_nonblocking():
    if os.name == 'nt':
        if msvcrt.kbhit():
            char = msvcrt.getwch()
            return char
        return None
    else:
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            return sys.stdin.readline().strip()
        return None

# =========================================================
#  헬퍼 함수
# =========================================================

def pwm_to_degree(pwm_value):
    """
    다이나믹셀 PWM(0~4095)을 각도(Degree)로 변환
    XL430/330 기준: 1 tick = 0.088도
    """
    # 0 ~ 4095 -> 0 ~ 360도
    return pwm_value * 0.088

def move_to_pos_blocking(target_full_pwm, robot, sim_robot, m, d, viewer, speed=6, timeout=8.0):
    start_time = time.time()
    curr = np.array(robot.read_position())
    target_full_pwm = np.array(target_full_pwm)

    while True:
        if time.time() - start_time > timeout:
            print("   [WARNING] 이동 타임아웃!")
            return False

        diff = target_full_pwm - curr
        if np.all(np.abs(diff) < 10):
            break

        step = np.clip(diff, -speed, speed)
        curr = (curr + step).astype(int)

        robot.set_goal_pos(curr)

        d.qpos[:6] = sim_robot._pwm2pos(curr) + JOINT_OFFSETS
        mujoco.mj_step(m, d)
        viewer.sync()

        cmd = get_command_nonblocking()
        if cmd == 'x': return False

        time.sleep(0.01)
    return True

def calc_radius_and_center(angle_diff,position,ee=0.05):
    distance=math.sqrt(position[0]**2+position[1]**2)+ee
    s=math.sin(angle_diff/2*math.pi/180)
    radius=distance*s/(1-s)
    cx=position[0]*(1+(radius+ee)/distance)
    cy=position[1]*(1+(radius+ee)/distance)
    center=cx,cy
    return radius, center

def find_safe_floor_j2(current_pose, target_j2, sim_robot, m, d):
    """안전 착륙 지점 계산 (시뮬레이션 예지)"""
    test_pose = current_pose.copy()
    start_j2 = test_pose[1]
    step = -10 if target_j2 < start_j2 else 10
    safe_j2 = start_j2

    for val in range(start_j2, target_j2, step):
        test_pose[1] = val
        d.qpos[:6] = sim_robot._pwm2pos(test_pose) + JOINT_OFFSETS
        mujoco.mj_kinematics(m, d)
        z_height = sim_robot.read_ee_pos(EE_SITE_NAME)[2]

        if z_height < (FLOOR_LIMIT + 0.02):
            return safe_j2
        safe_j2 = val

    return target_j2

def sweep_until_collision(direction, robot, sim_robot, m, d, viewer):
    curr_pwm = np.array(robot.read_position())
    joint_idx = 0

    print(f"   >>> 스윕 시작 ({'왼쪽' if direction > 0 else '오른쪽'})...")

    while True:
        curr_pwm[joint_idx] += PINCER_CONFIG['SWEEP_SPEED'] * direction

        if (direction > 0 and curr_pwm[joint_idx] > PINCER_CONFIG['LEFT_LIMIT_J0']) or \
           (direction < 0 and curr_pwm[joint_idx] < PINCER_CONFIG['RIGHT_LIMIT_J0']):
            return False, curr_pwm[joint_idx]

        robot.set_goal_pos(curr_pwm)
        d.qpos[:6] = sim_robot._pwm2pos(curr_pwm) + JOINT_OFFSETS
        mujoco.mj_step(m, d)
        viewer.sync()

        real_pos = np.array(robot.read_position())
        error = abs(curr_pwm[joint_idx] - real_pos[joint_idx])

        if error > PINCER_CONFIG['COLLISION_THRESH']:
            print(f"   !!! 충돌 감지! J0={real_pos[joint_idx]} (Err: {error})")
            curr_pwm[joint_idx] -= (PINCER_CONFIG['SWEEP_SPEED'] * direction * 5)
            robot.set_goal_pos(curr_pwm)
            time.sleep(0.5)
            return True, real_pos[joint_idx]

        cmd = get_command_nonblocking()
        if cmd == 'x': return False, None
        time.sleep(0.01)

def descend_and_detect_z(robot, sim_robot, m, d, viewer):
    curr_pwm = np.array(robot.read_position())
    joint_idx = 1 # J2 (Waist)

    print("   >>> 하강 시작 (DESCEND)...")

    target_val = Z_SEARCH_CONFIG['LOWEST_J2_PWM']
    direction = -1 if curr_pwm[joint_idx] > target_val else 1

    while True:
        curr_pwm[joint_idx] += Z_SEARCH_CONFIG['DESCEND_SPEED'] * direction

        if (direction < 0 and curr_pwm[joint_idx] < target_val) or \
           (direction > 0 and curr_pwm[joint_idx] > target_val):
            print("   -> 바닥 한계 도달 (충돌 없음)")
            return False, np.array(robot.read_position())

        robot.set_goal_pos(curr_pwm)
        d.qpos[:6] = sim_robot._pwm2pos(curr_pwm) + JOINT_OFFSETS
        mujoco.mj_step(m, d)
        viewer.sync()

        # --- 다관절 오차 감시 ---
        real_pos = np.array(robot.read_position())

        err_j2 = abs(curr_pwm[1] - real_pos[1])
        err_j3 = abs(curr_pwm[2] - real_pos[2])
        err_j4 = abs(curr_pwm[3] - real_pos[3])

        THRESH_MAIN = 30
        THRESH_SUB  = 60

        if (err_j2 > THRESH_MAIN) or (err_j3 > THRESH_SUB) or (err_j4 > THRESH_SUB):
            print(f"   !!! 충돌 감지! (J2:{err_j2}, J3:{err_j3}, J4:{err_j4})")
            for _ in range(50):
                # 꺾임 상쇄
                curr_pwm[1] += (Z_SEARCH_CONFIG['DESCEND_SPEED'])
                robot.set_goal_pos(curr_pwm)
                d.qpos[:6] = sim_robot._pwm2pos(curr_pwm) + JOINT_OFFSETS
                mujoco.mj_step(m, d)
                viewer.sync()
                time.sleep(0.01)
            
            real_pos = np.array(robot.read_position())
            collision_pose = real_pos.copy()
            time.sleep(1)
            print("위치 저장")
            # [긴급 회피] J2 최대 상승
            print("   >>> [긴급 회피] J2 최대 상승 (Lift Up)")
            safe_pwm = real_pos.copy()
            safe_pwm[1] = JOINT_LIMITS[1][1] # 2000

            move_to_pos_blocking(safe_pwm, robot, sim_robot, m, d, viewer, speed=5)
            return True, collision_pose

        cmd = get_command_nonblocking()
        if cmd == 'x': return False, None
        time.sleep(0.01)

def run_pincer_search(robot, sim_robot, m, d, viewer):
    print("\n=== [G] 핀서 탐색 (X축) 시작 ===")

    # 1. 초기화
    pwm = PINCER_CONFIG['INITIAL_POSE'].copy()
    if not move_to_pos_blocking(pwm, robot, sim_robot, m, d, viewer): return pwm

    # 2. 우측 이동
    pwm[0] = PINCER_CONFIG['RIGHT_LIMIT_J0']
    if not move_to_pos_blocking(pwm, robot, sim_robot, m, d, viewer): return pwm

    # 3. 팔 뻗기
    pwm = np.array([PINCER_CONFIG['RIGHT_LIMIT_J0'], 1234, 2310, 1745, 3000, 2009])
    if not move_to_pos_blocking(pwm, robot, sim_robot, m, d, viewer): return pwm

    # 4. 우 -> 좌 스윕
    found_r, angle_right = sweep_until_collision(1, robot, sim_robot, m, d, viewer)
    if not found_r: return np.array(robot.read_position())
    print(f"-> 오른쪽 경계: {angle_right}")

    # 5. 왼쪽 이동
    pwm = PINCER_CONFIG['INITIAL_POSE'].copy()
    if not move_to_pos_blocking(pwm, robot, sim_robot, m, d, viewer): return pwm
    pwm[0] = PINCER_CONFIG['LEFT_LIMIT_J0']
    if not move_to_pos_blocking(pwm, robot, sim_robot, m, d, viewer): return pwm

    pwm = np.array([PINCER_CONFIG['LEFT_LIMIT_J0'], 1234, 2310, 1745, 3000, 2009])
    if not move_to_pos_blocking(pwm, robot, sim_robot, m, d, viewer): return pwm

    # 6. 좌 -> 우 스윕
    found_l, angle_left = sweep_until_collision(-1, robot, sim_robot, m, d, viewer)
    if not found_l: return np.array(robot.read_position())
    print(f"-> 왼쪽 경계: {angle_left}")

    # 7. 중앙 정렬
    center_j0 = int((angle_right + angle_left) / 2)
    print(f"Step 7: 중앙({center_j0}) 정렬")

    pwm = PINCER_CONFIG['INITIAL_POSE'].copy()
    pwm[0] = angle_left
    if not move_to_pos_blocking(pwm, robot, sim_robot, m, d, viewer): return pwm
    pwm[0] = center_j0
    if not move_to_pos_blocking(pwm, robot, sim_robot, m, d, viewer): return pwm

    print("Step 8: 물체까지의 거리 측정 (접근 중...)")

    start_pwm = np.array(robot.read_position())

    # 목표 지점 설정 (J0는 현재 유지, 나머지는 확장)
    target_pwm = np.array([start_pwm[0], 1150, 2900, 1950, start_pwm[4]+200, start_pwm[5]])
    total_pwm_dist = np.linalg.norm(target_pwm - start_pwm)
    print(f"-> 전체 경로 길이(PWM Units): {total_pwm_dist:.2f}")

    # --- 접근 루프 설정 ---
    steps = 100
    collision_detected = False
    threshold = 100 # PINCER_CONFIG['COLLISION_THRESH']

    for i in range(steps):
        # 1. 선형 보간 (Linear Interpolation)
        ratio = (i + 1) / steps
        curr_target = start_pwm + (target_pwm - start_pwm) * ratio
        curr_target = curr_target.astype(int)

        robot.set_goal_pos(curr_target)

        time.sleep(0.05)

        real_pos = np.array(robot.read_position())
        errors = np.abs(curr_target - real_pos)
        max_error = np.max(errors)

        if max_error > threshold:
            print(f"!!! 충돌 감지! Step: {i}/{steps} (Max Error: {max_error})")
            collision_detected = True

            position=sim_robot.read_ee_pos(joint_name='joint6') # joint6의 위치
            distance=math.sqrt(position[0]**2+position[1]**2)
            print(f"base-joint6 최단거리:{distance}")

            angle_err=math.asin(0.02/distance) # 로봇팔 두께로 인한 각도 오차
            angle_diff=abs(pwm_to_degree(angle_left)-pwm_to_degree(angle_right)) # 로봇팔 각도차
            # angle_diff-=(angle_err*180/math.pi)

            # 물체의 반지름 및 중심 계산
            radius,center=calc_radius_and_center(angle_diff,position)
            print(f"물체 반지름: {radius}")
            print(f"물체 중점 좌표: {center}")
            
            # 마지막 충돌에서의 joint6 위치 저장
            global JOINT6_POSITION
            JOINT6_POSITION=position

            # 텐션 완화를 위한 백오프 (Back-off)
            back_off_pwm = curr_target - (target_pwm - start_pwm) * 0.1
            pwm = back_off_pwm.astype(int)



            # 최종 위치 명령
            robot.set_goal_pos(pwm)
            time.sleep(0.5)

            # --- 결과 계산 ---
            actual_dist = total_pwm_dist * ratio
            print(f"-> 측정된 거리 비율: {ratio:.2f} (1.0 = Max Reach)")
            print(f"-> 측정된 거리 값(PWM): {actual_dist:.2f}")
            break

        # 충돌이 없으면 현재 타겟 위치 업데이트
        pwm = curr_target

    if not collision_detected:
        print("-> 물체에 닿지 않고 최대 거리에 도달했습니다.")
        # 루프가 끝났으므로 마지막 위치 유지
        robot.set_goal_pos(target_pwm)
        pwm = target_pwm
        print(f"-> 측정된 거리 값(PWM): {total_pwm_dist:.2f}")

    print("=== X축 탐색 완료 ===")
    return pwm

def run_z_search(robot, sim_robot, m, d, viewer, target_j0):
    """
    [Z축 탐색] 회피 -> 전개 -> 스윙 -> 타격
    """
    print(f"\n=== [Z] 높이 탐색 시작 (Target: {target_j0}) ===")

    # [1] 회피 방향 결정 (Sidestep)
    mid_point = (PINCER_CONFIG['LEFT_LIMIT_J0'] + PINCER_CONFIG['RIGHT_LIMIT_J0']) / 2
    if target_j0 > mid_point:
        safe_side_j0 = PINCER_CONFIG['RIGHT_LIMIT_J0'] # 왼쪽 물체 -> 우측 회피
    else:
        safe_side_j0 = PINCER_CONFIG['LEFT_LIMIT_J0'] # 오른쪽 물체 -> 좌측 회피

    # [2] 웅크리기 & 회피
    print("Step Z-1: 회피 기동 (Sidestep)")
    safe_pose = PINCER_CONFIG['INITIAL_POSE'].copy()
    safe_pose[0] = safe_side_j0

    if not move_to_pos_blocking(safe_pose, robot, sim_robot, m, d, viewer, speed=5): return safe_pose
    curr_pwm = safe_pose.copy()

    # [3] 허공에서 최대 신전 (Expand)
    print("Step Z-2: 허공에서 최대 신전")

    curr_pwm[1] = JOINT_LIMITS[1][1] # J2 Max
    if not move_to_pos_blocking(curr_pwm, robot, sim_robot, m, d, viewer, speed=5): return curr_pwm

    curr_pwm[3] = JOINT_LIMITS[3][0] # J4 Max
    if not move_to_pos_blocking(curr_pwm, robot, sim_robot, m, d, viewer, speed=5): return curr_pwm

    curr_pwm[2] = 3150
    if not move_to_pos_blocking(curr_pwm, robot, sim_robot, m, d, viewer, speed=5): return curr_pwm

    # [4] 목표 지점으로 스윙 (Swing to Target)
    print(f"Step Z-3: 목표 지점({target_j0})으로 스윙")
    curr_pwm[0] = target_j0
    if not move_to_pos_blocking(curr_pwm, robot, sim_robot, m, d, viewer, speed=5): return curr_pwm

    # [5] J4 중간값 이동 (Chop Ready)
    j4_min, j4_max = JOINT_LIMITS[3]
    j4_middle = int((j4_min + j4_max) / 2)
    print(f"Step Z-4: J4 중간값 정렬 ({j4_middle})")
    curr_pwm[3] = j4_middle
    if not move_to_pos_blocking(curr_pwm, robot, sim_robot, m, d, viewer, speed=5): return curr_pwm

    safe_high_pose = curr_pwm.copy()

    # [6] 하강 및 충돌 감지 (The Chop)
    print("Step Z-5: 하강 및 충돌 감지")
    found, collision_pose = descend_and_detect_z(robot, sim_robot, m, d, viewer)

    if found:
        print("=== 충돌 감지 성공 ===")

        # 1. 시뮬레이터 FK로 손끝 높이 계산 (참고용)
        d.qpos[:6] = sim_robot._pwm2pos(collision_pose) + JOINT_OFFSETS
        mujoco.mj_kinematics(m, d)

        joint5_pos=sim_robot.read_ee_pos('joint5')
        joint2_pos=sim_robot.read_ee_pos('joint2')
        tan=(joint5_pos[2]-joint2_pos[2]-0.01)/(math.sqrt((joint5_pos[1]-joint2_pos[1])**2+(joint5_pos[0]-joint2_pos[0])**2))
        distance = math.sqrt(JOINT6_POSITION[0]**2 + JOINT6_POSITION[1]**2)
        height=(distance * tan + 0.06) * 100 # base와 joint2 높이차(6cm) 추가
        print(f"--------------------------------------------------")
        print(f"각도: {math.atan(tan)*180/math.pi}°")
        print(f"물체 높이: {height:.1f}cm")
        print(f"--------------------------------------------------")
        print("=== [탐색 종료] 안전 높이 복귀 완료 ===")

    else:
        print("=== 높이 측정 실패 (바닥 도달) ===")
        print("   >>> [복귀] 하강 전 안전 높이로 이동")

        # final_pose(바닥) -> safe_high_pose(위)
        move_to_pos_blocking(safe_high_pose, robot, sim_robot, m, d, viewer, speed=6)

    return np.array(robot.read_position())

# =========================================================
#  메인 루프
# =========================================================

print("=== 로봇 통합 제어 시스템 ===")
print(f"바닥 안전 높이: {FLOOR_LIMIT}m")
print("G: X축(너비) 탐색 -> 중앙 기억")
print("Z: Z축(높이) 탐색 (기억된 중앙으로 스윙)")
print("W/A/S/D...: 수동 제어")

# [중요] 마지막으로 찾은 중앙값을 기억하는 변수
LAST_FOUND_CENTER_J0 = None

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        cmd = get_command_nonblocking()
        key = ''
        if cmd is not None:
            if isinstance(cmd, bytes):
                try: key = cmd.decode('utf-8')
                except: key = ''
            else:
                key = cmd[0] if len(cmd) > 0 else ''

            if key == 'x':
                robot._disable_torque()
                print("finish")
                break

            if key == 'g':
                # G키: 탐색 수행 후 중앙값 저장
                final_pwm = run_pincer_search(robot, sim_robot, m, d, viewer)

                # 중앙값 추출 (final_pwm[0]가 중앙임)
                LAST_FOUND_CENTER_J0 = final_pwm[0]
                print(f"★ 중앙값 기억됨: {LAST_FOUND_CENTER_J0}")
                
                curr_pwm = final_pwm.copy()
                target_pwm = final_pwm.copy()
                key = ''

            elif key == 'z':
                # Z키: 기억된 중앙값이 없으면 현재 위치 사용
                target_j0 = LAST_FOUND_CENTER_J0 if LAST_FOUND_CENTER_J0 is not None else robot.read_position()[0]

                if LAST_FOUND_CENTER_J0 is None:
                    print("!!! 주의: G키 탐색 기록이 없어 현재 위치를 사용합니다.")

                final_pwm = run_z_search(robot, sim_robot, m, d, viewer, target_j0)
                curr_pwm = final_pwm.copy()
                target_pwm = final_pwm.copy()
                key = ''

        if teleop_active and key != '':
            next_target_pwm = target_pwm.copy()
            idx = -1; change = 0

            if key == 'a':   idx = 0; change = JOINT_SPEED * DIR[0]
            elif key == 'd': idx = 0; change = -JOINT_SPEED * DIR[0]
            elif key == 'e': idx = 1; change = JOINT_SPEED * DIR[1]
            elif key == 'c': idx = 1; change = -JOINT_SPEED * DIR[1]
            elif key == 'w': idx = 2; change = JOINT_SPEED * DIR[2]
            elif key == 's': idx = 2; change = -JOINT_SPEED * DIR[2]
            elif key == 'u': idx = 3; change = JOINT_SPEED * DIR[3]
            elif key == 'j': idx = 3; change = -JOINT_SPEED * DIR[3]
            elif key == 'i': idx = 4; change = JOINT_SPEED * DIR[4]
            elif key == 'k': idx = 4; change = -JOINT_SPEED * DIR[4]
            elif key == 'r': idx = 5; next_target_pwm[5] = GRIPPER_OPEN_PWM
            elif key == 't': idx = 5; next_target_pwm[5] = GRIPPER_CLOSE_PWM

            if idx != -1:
                if idx != 5: next_target_pwm[idx] += change

                min_limit, max_limit = JOINT_LIMITS[idx]
                next_val = next_target_pwm[idx]
                curr_val = target_pwm[idx]

                is_safe_joint = (min_limit <= next_val <= max_limit)
                is_escaping_min = (curr_val < min_limit) and (next_val > curr_val)
                is_escaping_max = (curr_val > max_limit) and (next_val < curr_val)

                if is_safe_joint or is_escaping_min or is_escaping_max:
                    if idx in [0, 4, 5]:
                        target_pwm[idx] = next_target_pwm[idx]
                        if idx == 5: print(f"[그리퍼] 동작")
                        else: print(f"[이동] J{idx+1} OK")
                    else:
                        curr_sim_pos = sim_robot._pwm2pos(curr_pwm) + JOINT_OFFSETS
                        d.qpos[:6] = curr_sim_pos
                        mujoco.mj_kinematics(m, d)
                        current_z = sim_robot.read_ee_pos(EE_SITE_NAME)[2]

                        sim_pos_prediction = sim_robot._pwm2pos(next_target_pwm) + JOINT_OFFSETS
                        d.qpos[:6] = sim_pos_prediction
                        mujoco.mj_kinematics(m, d)
                        future_z = sim_robot.read_ee_pos(EE_SITE_NAME)[2]

                        is_safe_height = (future_z >= FLOOR_LIMIT)
                        is_escaping_floor = (current_z < FLOOR_LIMIT) and (future_z > current_z)

                        if is_safe_height or is_escaping_floor:
                            target_pwm[idx] = next_target_pwm[idx]
                            status = "[탈출]" if is_escaping_floor else "[이동]"
                            print(f"{status} J{idx+1} PWM: {int(next_val)} | 높이: {future_z:.3f}m")
                        else:
                            print(f"[차단] 바닥 충돌 위험! 높이: {future_z:.3f}m")
                else:
                    print(f"[차단] J{idx+1} 한계 도달! ({int(next_val)})")

        if teleop_active:
            MAX_CHANGE = 10
            diff = target_pwm - curr_pwm
            step = np.clip(diff, -MAX_CHANGE, MAX_CHANGE)
            curr_pwm = (curr_pwm + step).astype(int)
            robot.set_goal_pos(curr_pwm)
            d.qpos[:6] = sim_robot._pwm2pos(curr_pwm) + JOINT_OFFSETS
            mujoco.mj_step(m, d)
            viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)