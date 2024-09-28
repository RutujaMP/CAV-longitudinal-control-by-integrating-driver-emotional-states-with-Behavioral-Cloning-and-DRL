import pandas as pd


def create_ramp_env(c):
    from automatic_vehicular_control.highway_ramp import RampEnv  # Import inside the function
    env = RampEnv(c)
    c.action_space = env.action_space
    c.observation_space = env.observation_space
    return env

psych_data = pd.read_csv('C:/automatic_vehicular_control/final_merged_data.csv')

def step(self, action=[]):
        c = self.c
        ts = self.ts
        max_dist = (c.premerge_distance + c.merge_distance) if c.global_obs else 100
        max_speed = c.max_speed

        # Assign driver data now that the simulation is running
        if not self.vehicle_driver_map:
            self.assign_driver_data(psych_data)

        #prev_rls = sorted(ts.vehicles.rl, key=lambda x: x.id)
        prev_rls = sorted([v for v in ts.vehicles.values() if v.type.id == 'rl'], key=lambda x: x.id)
        for rl, act in zip(prev_rls, action):
            if c.handcraft:
                continue
            else:
                # Only apply psychological states to human-driven vehicles
                if rl.type == 'human':
                    lead_vehicle = rl.leader()
                    if lead_vehicle:
                        lead_speed, dist_to_lead = lead_vehicle.speed, lead_vehicle.dist
                    else:
                        lead_speed, dist_to_lead = max_speed, max_dist

                    idm_acceleration = self.idm.calculate_acceleration(rl.speed, lead_speed, dist_to_lead)
                    idm_speed = rl.speed + idm_acceleration * ts.delta_t

                    # Retrieve psychological states for the specific driver
                    valence, arousal, emotion_state = self.get_driver_state(rl.id)

                    # Adjust IDM outputs based on psychological states
                    if valence is not None:
                        adjusted_accel, adjusted_speed = adjust_idm_output_with_psychology(idm_acceleration, idm_speed, valence, arousal, emotion_state, prediction_model)
                        ts.accel(rl, adjusted_accel)
                        rl.speed = adjusted_speed
                        print(f"[Human Vehicle {rl.id}] Psychological states - Valence: {valence}, Arousal: {arousal}, Emotion State: {emotion_state}")
                        print(f"Using predictions: Acceleration = {adjusted_accel}, Speed = {adjusted_speed}")
                    else:
                        # Default to IDM outputs if no psychological states are available
                        ts.accel(rl, idm_acceleration)
                        rl.speed = idm_speed
                        print(f"[Human Vehicle {rl.id}] Using IDM: Acceleration = {idm_acceleration}, Speed = {idm_speed}")
                else:
                    # Handle non-human-driven vehicles (e.g., RL-driven)
                    ts.accel(rl, act)
                    rl.speed += act * ts.delta_t

        super().step()

        obs, ids = [], []
        for veh in sorted([v for v in ts.vehicles.values() if v.type.id == 'rl'], key=lambda x: x.id):
            if veh.type == 'human':
                valence, arousal, emotion_state = self.get_driver_state(veh.id)
                emotion_mapping = {
                    'AD': [1, 0, 0, 0, 0, 0, 0],
                    'DD': [0, 1, 0, 0, 0, 0, 0],
                    'FD': [0, 0, 1, 0, 0, 0, 0],
                    'HD': [0, 0, 0, 1, 0, 0, 0],
                    'ND': [0, 0, 0, 0, 1, 0, 0],
                    'SAD': [0, 0, 0, 0, 0, 1, 0],
                    'SD': [0, 0, 0, 0, 0, 0, 1]
                }
                emotion_encoded = emotion_mapping.get(emotion_state, [0, 0, 0, 0, 0, 0, 0])
                obs.append({
                    'adjusted_speed': veh.speed,
                    'adjusted_accel': veh.acceleration,
                    'valence': valence,
                    'arousal': arousal,
                    'emotion_state': np.array(emotion_encoded)
                })
            else:
                # For RL vehicles, only include speed and acceleration in the observation
                obs.append({
                    'adjusted_speed': veh.speed,
                    'adjusted_accel': veh.acceleration,
                })
            ids.append(veh.id)

        obs = np.array([list(o.values()) for o in obs], dtype=np.float32)
        reward = len(ts.new_arrived) - c.collision_coef * len(ts.new_collided)
        self.latest_step_result = Namespace(obs=obs, id=ids, reward=reward)
        return Namespace(obs=obs, id=ids, reward=reward)
