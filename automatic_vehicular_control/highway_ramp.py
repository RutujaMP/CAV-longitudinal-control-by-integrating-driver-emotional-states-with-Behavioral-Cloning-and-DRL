import os
import subprocess
import numpy as np
import pandas as pd
from automatic_vehicular_control.exp import *
from automatic_vehicular_control.env import *
from automatic_vehicular_control.u import *
import time
import traci
import socket
import joblib
import gym
from gym.spaces import Box, Dict

# Load the Behavioral Cloning model for predicting adjustments
prediction_model = joblib.load('C:/automatic_vehicular_control/models/bc_prediction_model.joblib')

# Load the psychological states dataset
psych_data = pd.read_csv('C:/automatic_vehicular_control/final_merged_data.csv')

class IDM:
    def __init__(self, desired_speed=30.0, min_distance=2.0, time_headway=1.5, max_acceleration=1.5, comfortable_braking=2.0):
        self.desired_speed = desired_speed
        self.min_distance = min_distance
        self.time_headway = time_headway
        self.max_acceleration = max_acceleration
        self.comfortable_braking = comfortable_braking

    def calculate_acceleration(self, current_speed, lead_vehicle_speed, gap):
        desired_gap = self.min_distance + current_speed * self.time_headway + \
                      (current_speed * (current_speed - lead_vehicle_speed)) / (2 * np.sqrt(self.max_acceleration * self.comfortable_braking))
        
        acceleration = self.max_acceleration * (1 - (current_speed / self.desired_speed)**4 - (desired_gap / gap)**2)
        
        return acceleration



class RampEnv(Env):
    def __init__(self, c):
        super().__init__(c)

        self.idm = IDM()
       
        # Define action space
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Example action space

        # Define the observation spaces for human-driven and RL-driven vehicles
        self.human_observation_space = Dict({
            'adjusted_speed': Box(low=0, high=150, shape=(1,), dtype=np.float32),
            'adjusted_accel': Box(low=-10, high=10, shape=(1,), dtype=np.float32),
            'valence': Box(low=1, high=9, shape=(1,), dtype=np.float32),
            'arousal': Box(low=1, high=9, shape=(1,), dtype=np.float32),
            'emotion_state': Box(low=0, high=1, shape=(7,), dtype=np.float32)  # Assuming 7 possible states
        })

        self.rl_observation_space = Box(low=c.low, high=1, shape=(c._n_obs_non_human,), dtype=np.float32)

        from automatic_vehicular_control.env import current_vehicle_type

        print("current vehicle type",current_vehicle_type)

        if current_vehicle_type == 'human':
            print("entered the human observation space")
            self.observation_space = self.human_observation_space
        else:
            print("entered the other observation space")
            self.observation_space = self.rl_observation_space

        print("observation_space is set in highway_ramp",self.observation_space)

        self.sumo_paths = {
            'net': 'C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/net.xml',
            'nod': 'C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/nod.xml',
            'edg': 'C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/edg.xml',
            'con': 'C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/con.xml',
            'rou': 'C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/rou.xml',
            'add': 'C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/add.xml',
            'gui': 'C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/gui.xml'
        }

        # Prepare to handle multiple participants
       
        # self.assign_driver_data(psych_data)


    def find_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def start_sumo_gui(self):
        port = self.find_free_port()
        sumo_gui_cmd = [
            "sumo-gui",
            "--net-file", str(self.sumo_paths['net']),
            "--route-files", str(self.sumo_paths['rou']),
            "--additional-files", str(self.sumo_paths['add']),
            "--gui-settings-file", str(self.sumo_paths['gui']),
            "--collision.action", "remove",
            "--begin", "0",
            "--step-length", "0.5",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--no-warnings", "true",
            "--collision.check-junctions", "true",
            "--max-depart-delay", "0.5",
            "--random", "true",
            "--start", "true",
            "--remote-port", str(port)
        ]

        print("Starting SUMO-GUI with command:", " ".join(sumo_gui_cmd))
        self.sumo_process = subprocess.Popen(sumo_gui_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Monitor the SUMO-GUI process output for errors
        for _ in range(20):  # retry for 20 times (20*1s = 20s)
            try:
                traci.connect(port=port)
                print("Connected to TraCI server.")
                return
            except traci.exceptions.FatalTraCIError:
                print("Could not connect to TraCI server. Retrying in 1 second...")
                time.sleep(1)
            except subprocess.TimeoutExpired:
                continue

        raise RuntimeError("Could not connect to TraCI server")


    
    def def_sumo(self):
        c = self.c
        builder = NetBuilder()
        nodes = builder.add_nodes(Namespace(x=x, y=y) for x, y in [
            (0, 0),
            (c.premerge_distance, 0),
            (c.premerge_distance + c.merge_distance, 0),
            (c.premerge_distance + c.merge_distance + c.postmerge_distance, 0),
            (c.premerge_distance - 100 * np.cos(np.pi / 4), -100 * np.sin(np.pi / 4))
        ])
        builder.chain(nodes[[0, 1, 2, 3]], edge_attrs=[
            {}, {'numLanes': 2}, {}
        ], lane_maps=[
            {0: 1}, {0: 0, 1: 0}
        ], route_id='highway')
        builder.chain(nodes[[4, 1, 2, 3]], edge_attrs=[
            {}, {'numLanes': 2}, {}
        ], lane_maps=[
            {0: 0}, {0: 0, 1: 0}
        ], route_id='ramp')
        nodes, edges, connections, routes = builder.build()
        assert isinstance(nodes, list), "Nodes should be a list"
        assert isinstance(edges, list), "Edges should be a list"
        assert isinstance(connections, list), "Connections should be a list"
        assert isinstance(routes, list), "Routes should be a list"
        
        nodes[2].type = 'zipper'

        routes = E('routes',
            *routes,
            E('flow', **FLOW(f'f_highway', type='generic', route='highway', departSpeed=c.highway_depart_speed, vehsPerHour=c.highway_flow_rate)),
            E('flow', **FLOW(f'f_ramp', type='human', route='ramp', departSpeed=c.ramp_depart_speed, vehsPerHour=c.ramp_flow_rate))
        )

        # Define vehicle types directly using the correct parameters
        additional = E('additional',
            E('vType', id='generic', accel=1, decel=1.5, minGap=2),
            E('vType', id='rl', accel=1, decel=1.5, minGap=2),
            E('vType', id='human', accel=1, decel=1.5, minGap=2),
        )
        
        sumo_args = {'collision.action': COLLISION.remove}
        kwargs = self.sumo_def.save(nodes, edges, connections, routes, additional)
        kwargs['net'] = self.sumo_def.generate_net(**kwargs)
        kwargs['sumo_args'] = sumo_args
        self.sumo_def.sumo_cmd = self.sumo_def.generate_sumo(**kwargs)
        return kwargs

    # def step(self, action=[]):
    #     c = self.c
    #     ts = self.ts
    #     max_dist = (c.premerge_distance + c.merge_distance) if c.global_obs else 100
    #     max_speed = c.max_speed

    #     # Assign driver data now that the simulation is running
    #     if not self.vehicle_driver_map:
    #         self.assign_driver_data(psych_data)

    #     prev_rls = sorted([v for v in ts.vehicles.values() if v.type.id == 'rl'], key=lambda x: x.id)
    #     for rl, act in zip(prev_rls, action):
    #         if c.handcraft:
    #             continue
    #         else:
    #             if rl.type.id == 'human':
    #                 lead_vehicle = rl.leader()
    #                 if lead_vehicle:
    #                     lead_speed, dist_to_lead = lead_vehicle.speed, lead_vehicle.dist
    #                 else:
    #                     lead_speed, dist_to_lead = max_speed, max_dist

    #                 idm_acceleration = self.idm.calculate_acceleration(rl.speed, lead_speed, dist_to_lead)
    #                 idm_speed = rl.speed + idm_acceleration * ts.delta_t

    #                 # Retrieve psychological states for the specific driver
    #                 valence, arousal, emotion_state = self.get_driver_state(rl.id)

    #                 # Adjust IDM outputs based on psychological states
    #                 if valence is not None:
    #                     adjusted_accel, adjusted_speed = self.adjust_idm_output_with_psychology(idm_acceleration, idm_speed, valence, arousal, emotion_state, prediction_model)
    #                     ts.accel(rl, adjusted_accel)
    #                     rl.speed = adjusted_speed
    #                     print(f"[Human Vehicle {rl.id}] Psychological states - Valence: {valence}, Arousal: {arousal}, Emotion State: {emotion_state}")
    #                     print(f"Using predictions: Acceleration = {adjusted_accel}, Speed = {adjusted_speed}")
    #                 else:
    #                     ts.accel(rl, idm_acceleration)
    #                     rl.speed = idm_speed
    #                     print(f"[Human Vehicle {rl.id}] Using IDM: Acceleration = {idm_acceleration}, Speed = {idm_speed}")
    #             else:
    #                 ts.accel(rl, act)
    #                 rl.speed += act * ts.delta_t

    #     super().step()

    #     obs, ids = [], []
    #     reward = len(ts.new_arrived) - c.collision_coef * len(ts.new_collided)  # Initial reward computation
        
    #     # Add reward shaping based on emotional states for human-driven vehicles
    #     for veh in sorted([v for v in ts.vehicles.values() if v.type.id == 'rl'], key=lambda x: x.id):
    #         if veh.type.id == 'human':
    #             valence, arousal, emotion_state = self.get_driver_state(veh.id)
                
    #             # Example Reward Shaping Terms:
    #             if valence is not None:
    #                 # Reward higher valence (happier states)
    #                 reward += c.valence_coef * (valence - 5)  # Assuming 5 is the neutral midpoint

    #                 # Penalize higher arousal (indicating stress)
    #                 reward -= c.arousal_coef * (arousal - 5)  # Assuming 5 is the neutral midpoint

    #                 # Adjust reward based on categorical emotional states
    #                 emotion_bonus = self.get_emotion_bonus(emotion_state)
    #                 reward += emotion_bonus

    #             obs.append({
    #                 'adjusted_speed': veh.speed,
    #                 'adjusted_accel': veh.acceleration,
    #                 'valence': valence,
    #                 'arousal': arousal,
    #                 'emotion_state': self.encode_emotion_state(emotion_state)
    #             })
    #         else:
    #             obs.append({
    #                 'adjusted_speed': veh.speed,
    #                 'adjusted_accel': veh.acceleration,
    #             })
    #         ids.append(veh.id)

    #     obs = np.array([list(o.values()) for o in obs], dtype=np.float32)
    #     self.latest_step_result = Namespace(obs=obs, id=ids, reward=reward)
    #     return Namespace(obs=obs, id=ids, reward=reward)

    def step(self, action=[]):
        c = self.c
        ts = self.ts
        max_dist = (c.premerge_distance + c.merge_distance) if c.global_obs else 100
        max_speed = c.max_speed

        # Ensure vehicle types are properly defined
        try:
            human_type = ts.types.human
        except KeyError:
            human_type = ts.types.add(id='human', accel=1, decel=1.5, tau=1.0, minGap=2, maxSpeed=30, speedFactor=1.0, speedDev=0.1, impatience=0.5, delta=4, carFollowModel='IDM', sigma=0.2)
        rl_type = ts.types.rl

        # Assign driver data now that the simulation is running
        if not self.vehicle_driver_map:
            self.assign_driver_data(psych_data)

        prev_rls = sorted(rl_type.vehicles, key=lambda x: x.id)
        for rl, act in zip(prev_rls, action):
            if c.handcraft:
                route, edge, lane = rl.route, rl.edge, rl.lane
                leader, dist = rl.leader()
                level = 1
                if edge.id == 'e_n_0.0_n_400.0':
                    if rl.laneposition < 100:
                        leaders = list(rl.leaders())
                        if len(leaders) > 20:
                            level = 0
                        else:
                            level = (0.75 * np.sign(c.handcraft - rl.speed) + 1) / 2
                        ts.accel(rl, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))
                continue
            if not isinstance(act, (int, np.integer)):
                act = (act - c.low) / (1 - c.low)
            if c.act_type.startswith('accel'):
                level = act[0] if c.act_type == 'accel' else act / (c.n_actions - 1)
                ts.accel(rl, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))
            else:
                if c.act_type == 'continuous':
                    level = act[0]
                elif c.act_type == 'discretize':
                    level = min(int(act[0] * c.n_actions), c.n_actions - 1) / (c.n_actions - 1)
                elif c.act_type == 'discrete':
                    level = act / (c.n_actions - 1)
                ts.set_max_speed(rl, max_speed * level)

        super().step()

        # Initialize reward
        base_reward = len(ts.new_arrived) - c.collision_coef * len(ts.new_collided)
        reward = base_reward  # Start with the base reward

        # Collect observations and refine rewards for all vehicles
        obs = []
        ids = []
        
        for veh in sorted(rl_type.vehicles, key=lambda v: v.id):
            if veh.type.id == 'human':
                # Human vehicle logic
                lead_vehicle = veh.leader()
                if lead_vehicle:
                    lead_speed, dist_to_lead = lead_vehicle.speed, lead_vehicle.dist
                else:
                    lead_speed, dist_to_lead = max_speed, max_dist

                idm_acceleration = self.idm.calculate_acceleration(veh.speed, lead_speed, dist_to_lead)
                idm_speed = veh.speed + idm_acceleration * ts.delta_t

                # Retrieve psychological states for the specific driver
                valence, arousal, emotion_state = self.get_driver_state(veh.id)

                # Adjust IDM outputs based on psychological states
                if valence is not None:
                    adjusted_accel, adjusted_speed = self.adjust_idm_output_with_psychology(
                        idm_acceleration, idm_speed, valence, arousal, emotion_state, prediction_model
                    )
                    ts.accel(veh, adjusted_accel)
                    veh.speed = adjusted_speed

                    # Reward shaping based on psychological states for human vehicles
                    reward += c.valence_coef * (valence - 5)  # Reward for higher valence (happier states)
                    reward -= c.arousal_coef * (arousal - 5)  # Penalize higher arousal (indicating stress)
                    emotion_bonus = self.get_emotion_bonus(emotion_state)
                    reward += emotion_bonus

                    obs.append([
                        veh.speed,  # adjusted_speed
                        adjusted_accel,  # adjusted_accel
                        valence,
                        arousal,
                        *self.encode_emotion_state(emotion_state)
                    ])
                else:
                    ts.accel(veh, idm_acceleration)
                    veh.speed = idm_speed

                    obs.append([
                        veh.speed,  # adjusted_speed
                        idm_acceleration,  # adjusted_accel
                        0, 0, 0, 0, 0, 0, 0  # Placeholder for valence, arousal, and emotion_state
                    ])
            else:
                # Non-human vehicle logic (retaining the previous functionality)
                speed, edge, lane = veh.speed, veh.edge, veh.lane
                merge_dist = max_dist

                lead_speed = follow_speed = other_speed = 0
                other_follow_dist = other_merge_dist = lead_dist = follow_dist = max_dist

                leader, dist = veh.leader()
                if leader: lead_speed, lead_dist = leader.speed, dist

                follower, dist = veh.follower()
                if follower: follow_speed, follow_dist = follower.speed, dist

                if c.global_obs:
                    jun_edge = edge.next(ts.routes.highway)
                    while jun_edge and not (len(jun_edge.lanes) == 2 and jun_edge.lanes[0].get('junction')):
                        jun_edge = jun_edge.next(ts.routes.highway)
                    if jun_edge:
                        merge_dist = lane.length - veh.laneposition
                        next_edge = edge.next(ts.routes.highway)
                        while next_edge is not jun_edge:
                            merge_dist += next_edge.length
                            next_edge = next_edge.next(ts.routes.highway)

                        other_lane = jun_edge.lanes[0]
                        for other_veh, other_merge_dist in other_lane.prev_vehicles(0, route=ts.routes.ramp):
                            other_speed = other_veh.speed
                            break
                    obs.append([merge_dist, speed, lead_dist, lead_speed, follow_dist, follow_speed, other_merge_dist, other_speed])
                else:
                    next_lane = lane.next(ts.routes.highway)
                    if next_lane and next_lane.get('junction'):
                        if len(edge.lanes) == 2:
                            other_lane = edge.lanes[0]
                            pos = veh.laneposition
                            for other_veh, other_follow_dist in other_lane.prev_vehicles(pos, route=ts.routes.ramp):
                                other_speed = other_veh.speed
                                break
                    obs.append([speed, lead_dist, lead_speed, follow_dist, follow_speed, other_follow_dist, other_speed])

            ids.append(veh.id)

        # Normalize and clip observations differently for human and non-human vehicles
        if obs:
            obs = np.array(obs)
            if obs.shape[1] == c._n_obs_human:
                obs = obs / ([max_speed, max_speed, 9, 9] + [1] * 7)
            else:
                obs = obs / ([*lif(c.global_obs, max_dist), max_speed] + [max_dist, max_speed] * 3)
            obs = np.clip(obs, 0, 1).astype(np.float32) * (1 - c.low) + c.low
        else:
            obs = np.zeros((1, c._n_obs_human if 'human' in ts.types else c._n_obs_non_human), dtype=np.float32)

        # Final return
        self.latest_step_result = Namespace(obs=obs, id=ids, reward=reward)
        return Namespace(obs=obs, id=ids, reward=reward)



    # Helper function to provide a bonus/penalty based on the categorical emotional state
    def get_emotion_bonus(self, emotion_state):
        emotion_bonus_mapping = {
            'AD': -1.0,  # Penalty for "Anger"
            'SAD': -0.5,  # Penalty for "Sadness"
            'FD': -0.7,  # Penalty for "Fear"
            'DD': -0.6,  # Penalty for "Disgust"
            'SD': 0.2,  # Small bonus for "Surprise"
            'HD': 1.0,  # Bonus for "Happiness"
            'ND': 0.0  # Neutral for "Neutral"
        }
        return emotion_bonus_mapping.get(emotion_state, 0)


class Ramp(Main):
 
    def create_env(self, c):
        return RampEnv(c)

    @property
    def observation_space(c):
        

        return Dict({
            'adjusted_speed': Box(low=0, high=150, shape=(1,), dtype=np.float32),  # Example ranges
            'adjusted_accel': Box(low=-10, high=10, shape=(1,), dtype=np.float32),
            'valence': Box(low=1, high=9, shape=(1,), dtype=np.float32),
            'arousal': Box(low=1, high=9, shape=(1,), dtype=np.float32),
            'emotion_state': Box(low=0, high=1, shape=(7,), dtype=np.float32)  # Updated shape for 7 possible states
        })

    @property
    def action_space(c):
        return Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Example action space

    def on_rollout_end(c, rollout, stats, ii=None, n_ii=None):
        log = c.get_log_ii(ii, n_ii)
        step_obs_ = rollout.obs
        step_obs = step_obs_[:-1]

        ret, _ = calc_adv(rollout.reward, c.gamma)

        n_veh = np.array([len(o) for o in step_obs])
        step_ret = [[r] * nv for r, nv in zip(ret, n_veh)]
        rollout.update(obs=step_obs, ret=step_ret)

        step_id_ = rollout.pop('id')
        id = np.concatenate(step_id_[:-1])
        id_unique = np.unique(id)

        reward = np.array(rollout.pop('reward'))

        log(**stats)
        log(reward_mean=reward.mean(), reward_sum=reward.sum())
        log(n_veh_step_mean=n_veh.mean(), n_veh_step_sum=n_veh.sum(), n_veh_unique=len(id_unique))
        return rollout

if __name__ == '__main__':
    c = Ramp.from_args(globals(), locals()).setdefaults(
        warmup_steps=100,
        horizon=1000,
        n_steps=20,
        step_save=5,

        premerge_distance=400,
        merge_distance=100,
        postmerge_distance=30,
        av_frac=0.1,
        sim_step=0.5,
        max_speed=30,
        highway_depart_speed=10,
        ramp_depart_speed=0,
        highway_flow_rate=1500,
        ramp_flow_rate=300,
        global_obs=False,
        handcraft=False,

        generic_type='default',
        speed_mode=SPEED_MODE.all_checks,
        collision_coef=5,  # If there's a collision, it always involves an even number of vehicles

        act_type='accel_discrete',
        max_accel=1,
        max_decel=1.5,
        n_actions=3,
        low=-1,

        render=True,

        alg=PG,
        lr=1e-3,

        gamma=0.99,
        adv_norm=False,
        batch_concat=True,
    )
    #c._n_obs = c.global_obs + 1 + 2 + 2 + 7  # Adjusted observation space size
    c._n_obs_human = c.global_obs + 1 + 2 + 2 + 2 + 1 + 1 + 7
    c._n_obs_non_human = c.global_obs + 1 + 2 + 2 + 2


    c.run()
