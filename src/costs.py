import numpy as np
from flwr.common import logger
import logging
from typing import List, Tuple

def measure_communication_cost(downlink_size: int, uplink_size: int, num_clients: int) -> Tuple[List[int], int]:
    rho = 5e-9 # cost of receiving one bit J/bit
    beta1 = 1e-8 # cost of processing 1 bit of data, 1e-8 J/bit
    beta2 = 1e-7 # cost of transferring 1 bit of data, 1e-10 J/bit/m^3
    alpha = 3 # path loss exponent
    # Simulate the locations of the clients
    radius = 100  # radius of the circle
    angles = np.random.uniform(0, 2 * np.pi, num_clients)
    distances = np.random.normal(radius / 2, radius / 4, num_clients)

    # Ensure distances are within the circle
    distances = np.clip(distances, 0, radius)

    # Calculate the x and y coordinates of the clients
    x_coords = distances * np.cos(angles)
    y_coords = distances * np.sin(angles)

    # Calculate the distance from each client to the server at the origin
    client_distances = np.sqrt(x_coords**2 + y_coords**2)
    logger.log(logging.INFO, f"Clients distances: {client_distances}, downlink_size: {downlink_size}, uplink_size: {uplink_size}")
    
    sending_ery_per_bit = beta1 + beta2 * client_distances**alpha
    comm_cost = rho * downlink_size + sending_ery_per_bit * uplink_size
    avg_comm_cost = np.sum(comm_cost) / num_clients
    logger.log(logging.INFO, f"Communication cost: {comm_cost}, avg_comm_cost: {avg_comm_cost}")
    return comm_cost.tolist(), avg_comm_cost
    

def measure_computation_cost(computation_time: np.ndarray) -> np.ndarray:
    # Assume the cost of computation is 1e-6 J/flop
    beta3 = 1e-6
    # Assume the IoT device needs 600x more time compared to GPUs
    iot_speed_ratio = 800
    if isinstance(computation_time, np.ndarray):
        comp_cost = beta3 * computation_time * iot_speed_ratio
    elif isinstance(computation_time, list):
        comp_cost = [beta3 * time * iot_speed_ratio for time in computation_time]
    elif isinstance(computation_time, float):
        comp_cost = beta3 * computation_time * iot_speed_ratio
    logger.log(logging.INFO, f"Computation cost: {comp_cost}")
    return comp_cost


def cost_efficiency(total_cost, accuracy):
    return accuracy / total_cost