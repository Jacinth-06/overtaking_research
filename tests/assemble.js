const fs = require('fs');

const v1 = fs.readFileSync('version1.py', 'utf8');
const lines = v1.split('\n');

const startIdx = lines.findIndex(l => l.startsWith('def control_loop(car: JetRacer):'));
const controlLoopAndBeyond = lines.slice(startIdx).join('\n');

const oldLogic = `        # 3. STATE MACHINE (Strictly IF/ELIF, NO WHILE LOOPS)
        if s_copy["enabled"]:
            if autonomy_state == "FOLLOW":
                # Track nominal lane width
                if left_found and right_found and lane_width > 50:
                    old_lw = pid_state.get("nominal_lane_width", lane_width)
                    pid_state["nominal_lane_width"] = 0.95 * old_lw + 0.05 * lane_width

                if lidar_blocked:
                    autonomy_state = "OVERTAKING"
                    pid_state["crossing_phase"] = 1
                    pid_state.pop("phase_debounce_time", None) # Clear any old timers
                    print(f"\\n[STATE CHANGE] -> OVERTAKING. Obstacle at {lidar_closest}mm", flush=True)
                    car.steer(0.8)                              
                    car.forward(s_copy["speed"])
                else:
                    car.steer(steer)
                    car.forward(s_copy["speed"])
                    
            elif autonomy_state == "OVERTAKING":
                car.steer(0.8)                                  
                car.forward(s_copy["speed"])
                
                phase = pid_state.get("crossing_phase", 1)
                now = time.time()
                
                # PHASE 1: Wait for original left lane to disappear STABLY for 2 full seconds
                if phase == 1:
                    if not left_found:
                        if "phase_debounce_time" not in pid_state:
                            pid_state["phase_debounce_time"] = now
                        elif now - pid_state["phase_debounce_time"] >= 2.0:
                            pid_state["crossing_phase"] = 2
                            pid_state.pop("phase_debounce_time", None) 
                            print("\\n[STATE CHANGE] OVERTAKING Phase 1 -> 2 (CONFIRMED lost old left lane)", flush=True)
                    else:
                        pid_state.pop("phase_debounce_time", None) 
                        
                # PHASE 2: Wait for original right lane to cross center to become new left lane for 1 full second
                elif phase == 2:
                    if left_found:
                        if "phase_debounce_time" not in pid_state:
                            pid_state["phase_debounce_time"] = now
                        elif now - pid_state["phase_debounce_time"] >= 0.5:
                            pid_state["crossing_phase"] = 3
                            pid_state.pop("phase_debounce_time", None)
                            print("\\n[STATE CHANGE] OVERTAKING Phase 2 -> 3 (CONFIRMED old right lane became new left)", flush=True)
                    else:
                        pid_state.pop("phase_debounce_time", None)
                        
                # PHASE 3: Wait for a brand new right lane to appear stably for 1 full second (LANE WIDTH CHECK REMOVED)
                elif phase == 3:
                    if right_found:
                        if "phase_debounce_time" not in pid_state:
                            pid_state["phase_debounce_time"] = now
                        elif now - pid_state["phase_debounce_time"] >= 0.5:
                            autonomy_state = "CHECKING"
                            pid_state["crossing_phase"] = 1
                            pid_state.pop("phase_debounce_time", None)
                            print("\\n[STATE CHANGE] -> CHECKING. Switched to right lane safely.", flush=True)
                            pid_state["integral"] = 0.0
                            pid_state["last_error"] = 0.0
                    else:
                        pid_state.pop("phase_debounce_time", None)
                    
            elif autonomy_state == "CHECKING":
                car.steer(steer)
                car.forward(s_copy["speed"])
                # Wait until safe left distance (obstacle crossed)
                phase = pid_state.get("crossing_phase", 1)
                if phase == 1:
                    # phase 1: detect obstacle on the left
                    if lidar_closest_left > 0.0 and lidar_closest_left < 600.0:
                        pid_state["crossing_phase"] = 2
                        print(f"\\n[STATE CHANGE] CHECKING -> Phase 2 (obstacle {lidar_closest_left}mm to left)", flush=True)
                elif phase == 2:
                    # phase 2: obstacle clears
                    if lidar_closest_left == 0.0 or lidar_closest_left > 300.0:
                        autonomy_state = "RECOVERY"
                        pid_state["crossing_phase"] = 1
                        pid_state.pop("phase_debounce_time", None) 
                        print("\\n[STATE CHANGE] -> RECOVERY. Left side clear.", flush=True)
                    
            elif autonomy_state == "RECOVERY":
                # Check if a new obstacle appeared ahead OR if the left obstacle returned
                is_front_blocked = lidar_blocked
                is_left_blocked  = (0.0 < lidar_closest_left < 300.0)

                if is_front_blocked or is_left_blocked:
                    # NOT SAFE TO RETURN: Override hard turn and follow current lane line
                    car.steer(steer)
                    car.forward(s_copy["speed"])
                    pid_state["crossing_phase"] = 1
                    pid_state.pop("phase_debounce_time", None)
                else:
                    # SAFE: Execute normal recovery hard turn
                    car.steer(-0.8)                                 
                    car.forward(s_copy["speed"])
                    
                    phase = pid_state.get("crossing_phase", 1)
                    now = time.time()
                    
                    # PHASE 1: Wait for right lane to disappear STABLY for 2 full seconds
                    if phase == 1:
                        if not right_found:
                            if "phase_debounce_time" not in pid_state:
                                pid_state["phase_debounce_time"] = now
                            elif now - pid_state["phase_debounce_time"] >= 2.0:
                                pid_state["crossing_phase"] = 2
                                pid_state.pop("phase_debounce_time", None)
                                print("\\n[STATE CHANGE] RECOVERY Phase 1 -> 2 (CONFIRMED lost old right lane)", flush=True)
                        else:
                            pid_state.pop("phase_debounce_time", None)
                            
                    # PHASE 2: Wait for left lane to cross center to become new right lane for 1 full second
                    elif phase == 2:
                        if right_found:
                            if "phase_debounce_time" not in pid_state:
                                pid_state["phase_debounce_time"] = now
                            elif now - pid_state["phase_debounce_time"] >= 0.5:
                                pid_state["crossing_phase"] = 3
                                pid_state.pop("phase_debounce_time", None)
                                print("\\n[STATE CHANGE] RECOVERY Phase 2 -> 3 (CONFIRMED old left lane became new right)", flush=True)
                        else:
                            pid_state.pop("phase_debounce_time", None)
                            
                    # PHASE 3: Wait for a brand new left lane to appear stably for 1 full second (LANE WIDTH CHECK REMOVED)
                    elif phase == 3:
                        if left_found:
                            if "phase_debounce_time" not in pid_state:
                                pid_state["phase_debounce_time"] = now
                            elif now - pid_state["phase_debounce_time"] >= 0.5:
                                autonomy_state = "FOLLOW"
                                pid_state["crossing_phase"] = 1
                                pid_state.pop("phase_debounce_time", None)
                                print("\\n[STATE CHANGE] -> FOLLOW. Back in original lane safely.", flush=True)
                                pid_state["integral"] = 0.0
                                pid_state["last_error"] = 0.0
                        else:
                            pid_state.pop("phase_debounce_time", None)
                    
            # Set telemetry steering monitor value based on current operational mode
            if autonomy_state == "OVERTAKING": 
                steer = 0.8
            elif autonomy_state == "RECOVERY": 
                if (lidar_blocked or (0.0 < lidar_closest_left < 300.0)):
                    pass 
                else:
                    steer = -0.8`;

const newLogic = `        import math
        
        # 3. Odometry tracking for lateral position
        dt = max(now - last_time, 0.001)
        last_time = now
        
        yaw_rate_rad = math.radians(imu_gz)
        yaw += yaw_rate_rad * dt
        
        vy = enc_speed * math.sin(yaw)
        pos_y += vy * dt

        # 4. STATE MACHINE
        if s_copy["enabled"]:
            if autonomy_state == "FOLLOW":
                if lidar_blocked:
                    autonomy_state = "OVERTAKING"
                    pid_state["start_enc_dist"] = enc_dist
                    pid_state["start_pos_y"] = pos_y
                    # Distance over which to lane change, based on lidar distance
                    pid_state["lane_change_dist"] = max(lidar_closest / 1000.0 - 0.2, 0.5) 
                    print(f"\\n[STATE CHANGE] -> OVERTAKING. Obstacle at {lidar_closest}mm", flush=True)
                else:
                    car.steer(steer)
                    car.forward(s_copy["speed"])
                    
            elif autonomy_state == "OVERTAKING":
                s = enc_dist - pid_state.get("start_enc_dist", enc_dist)
                D = pid_state.get("lane_change_dist", 1.0)
                W = 0.28
                
                if s < D:
                    s_ratio = s / D
                    target_y = pid_state.get("start_pos_y", 0.0) + W * (10 * (s_ratio)**3 - 15 * (s_ratio)**4 + 6 * (s_ratio)**5)
                else:
                    target_y = pid_state.get("start_pos_y", 0.0) + W
                
                traj_error = target_y - pos_y
                
                pid_traj["integral"] += traj_error * dt
                pid_traj["integral"] = Math.max(-1.0, Math.min(1.0, pid_traj["integral"])) // Need python syntax
                pid_traj["integral"] = max(-1.0, min(1.0, pid_traj["integral"]))
                derivative = (traj_error - pid_traj["last_error"]) / dt
                pid_traj["last_error"] = traj_error
                
                steer_cmd = s_copy["kp"] * traj_error + s_copy["ki"] * pid_traj["integral"] + s_copy["kd"] * derivative
                steer_cmd = max(-1.0, min(1.0, steer_cmd))
                
                car.steer(steer_cmd)
                car.forward(s_copy["speed"])
                steer = steer_cmd`;

let codeStr = controlLoopAndBeyond.replace(oldLogic, newLogic);

const oldInit = `    cap = open_camera()
    fps_counter, fps_time = 0, time.time()
    frame_idx = 0
    print("[loop] Control loop started")`;

const newInit = `    cap = open_camera()
    fps_counter, fps_time = 0, time.time()
    frame_idx = 0
    
    yaw = 0.0
    pos_y = 0.0
    last_time = time.time()
    pid_traj = {"integral": 0.0, "last_error": 0.0}
    
    print("[loop] Control loop started")`;

codeStr = codeStr.replace(oldInit, newInit);

// process_frame signature update is not needed because it is not in the control_loop definition

fs.appendFileSync('version3.py', '\n' + codeStr);
console.log('Done!');
