import numpy as np


def convert_flow3d_to_experimental_pv(flow3d_pv):
    """
    Convert a FLOW-3D process condition key to the experimental format.

    Args:
        flow3d_pv (str): Key in the form "power280_velocity450", with velocity in mm/s.

    Returns:
        str: Key in the form "280_0.45", with velocity in m/s.
    """
    power = flow3d_pv.split('_')[0].split('power')[-1]
    velocity = int(flow3d_pv.split('_')[1].split('velocity')[-1])
    velocity_conversion = velocity/1000 # convert to m/s
    updated_power_velocity = f"{power}_{velocity_conversion}"
    return updated_power_velocity

def convert_ET_to_experimental_pv(ET_pv):
    """
    Convert an Eagar-Tsai process condition key to the experimental format.

    Args:
        ET_pv (str): Key in the form "power_280_velocity450", with velocity in mm/s.

    Returns:
        str: Key in the form "280.0_0.45", with velocity in m/s.
    """
    power  = float(ET_pv.split('power_')[-1].split('_')[0])
    velocity = float(ET_pv.split('_')[-1].split('velocity')[-1])
    velocity_conversion = velocity/1000 # convert to m/s
    updated_power_velocity = f"{power}_{velocity_conversion}"
    return updated_power_velocity

def parse_process_param(pv_string):
    """
    Split a process condition key into power and velocity.

    Args:
        pv_string (str): Either a FLOW-3D key, "power280_velocity450", or an experimental key,
            "280_0.45".

    Returns:
        tuple: (power, velocity). FLOW-3D keys give ints with velocity in mm/s. Experimental keys
            give floats with velocity in m/s.
    """


    if 'power' in pv_string and 'velocity' in pv_string:
        # FLOW-3D dataset
        power = int(pv_string.split('_')[0].split('power')[1])
        velocity = int(pv_string.split('_')[1].split('velocity')[1])
    
    else:
        power = float(pv_string.split('_')[0])
        velocity = float(pv_string.split('_')[1])
    return power, velocity


def energy_density(power, velocity):
    """
    Compute the linear energy density power divided by velocity in J/mm.

    Args:
        power (float or array): Laser power in W.
        velocity (float or array): Scan velocity in m/s. FLOW-3D keys give velocity in mm/s and must be
            converted first.

    Returns:
        numpy.ndarray: Energy density, computed elementwise.
    """
    return np.asarray(power, dtype = float) / (1000 * np.asarray(velocity, dtype = float))
