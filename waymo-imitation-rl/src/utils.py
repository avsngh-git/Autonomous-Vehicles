import numpy as np

class PIDController:
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.1):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.prev_error = 0
        self.integral = 0

    def get_control(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.k_p * error + self.k_i * self.integral + self.k_d * derivative
        self.prev_error = error
        return output

def get_expert_action(vehicle, target_waypoint):
    """
    Calculates the steering/throttle needed for 'vehicle' to hit 'target_waypoint'.
    """
    vehicle_pos = vehicle.position
    vehicle_heading = vehicle.heading_theta
    
    target_vector = target_waypoint - vehicle_pos
    target_heading = np.arctan2(target_vector[1], target_vector[0])
    
    heading_error = target_heading - vehicle_heading
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
    
    # PID for Steering
    steering = np.clip(heading_error * 1.5, -1.0, 1.0)
    
    # Simple Throttle Logic
    throttle = 0.6 if abs(steering) < 0.3 else 0.3
    
    return np.array([steering, throttle])
