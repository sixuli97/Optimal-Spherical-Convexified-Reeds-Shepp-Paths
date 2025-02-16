"""
Script Name: CRS_on_sphere_opt.py
Author: Sixu Li
Email: sixuli@tamu.edu
Created: 2025-01-08
Description:
    This script provides functions to generate the time-optimal convexified Reeds-Shepp path on a sphere.
"""

import numpy as np
import math
from math import cos as cos
from math import sin as sin
from math import sqrt as sqrt
from math import acos as acos
from math import atan2 as atan2
from math import atan as atan
from math import pi as pi
from plotting_class import plotting_functions
from matplotlib.cm import get_cmap



def trans_2_standard(max_turn_rate, initial_config, terminal_config, sphere_radius=1):
    """
    Scale the maximum turning rate of the CRS car to a unit-sphere problem.
    Transform the initial configuration and the terminal configuration in SO(3) such that the initial configuration becomes the identity matrix I_3.

    Parameters:
        sphere_radius: scalar representing the radius of the sphere.
        max_turn_rate: scalar representing the maximum turning rate.
        initial_config: 3x3 rotation matrix in SO(3) representing the initial configuration.
        terminal_config: 3x3 rotation matrix in SO(3) representing the terminal configuration.

    Returns:
        U_max: scalar representing scaled maximum turning rate.
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
    """
    # scale maximum turning rate and calculate scaled turning radius
    U_max = max_turn_rate * sphere_radius
    r_min = 1 / sqrt(1 + U_max**2)

    # Compute the transformation matrix that maps the initial configuration to I_3
    transformation = np.linalg.inv(initial_config)

    # Transform the terminal configuration
    Alpha = np.dot(transformation, terminal_config)

    # Return the identity matrix for the initial configuration and the transformed terminal configuration
    return U_max, r_min, Alpha

def trans_2_standard_sym(max_turn_rate, initial_config, terminal_config, sphere_radius=1):
    """
    This function is for the symmetric forms (string inverse) of paths in the sufficient list.
    Scale the maximum turning rate of the CRS car to a unit-sphere problem.
    Transform the initial configuration and the terminal configuration in SO(3) such that the terminal configuration becomes the identity matrix I_3.

    Parameters:
        sphere_radius: scalar representing the radius of the sphere.
        max_turn_rate: scalar representing the maximum turning rate.
        initial_config: 3x3 rotation matrix in SO(3) representing the initial configuration.
        terminal_config: 3x3 rotation matrix in SO(3) representing the terminal configuration.

    Returns:
        U_max: scalar representing scaled maximum turning rate.
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
    """
    # scale maximum turning rate and calculate scaled turning radius
    U_max = max_turn_rate * sphere_radius
    
    r_min = 1 / sqrt(1 + U_max**2)

    # Compute the transformation matrix that maps the initial configuration to I_3
    transformation = np.linalg.inv(terminal_config)

    # Transform the terminal configuration
    Alpha = np.dot(transformation, initial_config)

    # Return the identity matrix for the initial configuration and the transformed terminal configuration
    return U_max, r_min, Alpha


def Rotation_cal(type, phi, U_max):
    """
    Calculate the transformation matrix of a segment with specified type and angle in radians.

    Parameters:
        type: string representing the segment type. "+" and "-" represent forward and backward, respectively. "L","R", "G" represent left, right, and great circle, respectively.
        phi: scalar representing the angle in radians of the segment.
        U_max: scalar representing the scaled maximum turning rate.

    Returns:
        R: 3x3 transformation matrix that transforms the initial config of the segment to its terminal config.
    """

    if type == "G+":
        v = 1
        u = 0
    elif type == "L+":
        v = 1
        u = U_max
    elif type == "R+":
        v = 1
        u = -U_max
    elif type == "G-":
        v = -1
        u = 0
    elif type == "L-":
        v = -1
        u = U_max
    elif type == "R-":
        v = -1
        u = -U_max
    elif type == "L0":
        v = 0
        u = U_max
    elif type == "R0":
        v = 0
        u = -U_max


    r = 1 / sqrt(v**2 + u**2)

    omega_hat = (
        np.array(
            [
                [0, -v * r, 0],
                [v * r, 0, -u * r],
                [0, u * r, 0],
            ]
        )
    ).T

    R = (
        np.eye(3)
        + sin(phi) * omega_hat
        + (1 - cos(phi)) * np.matmul(omega_hat, omega_hat)
    ).T

    return R


def SO3_distance_check(R1, R2):
    """
    Compute the geodesic distance between two rotation matrices in SO(3) and check if less than threshold epsilon.

    Parameters:
        R1, R2: 3x3 rotation matrices in SO(3).

    Returns:
        bool: True if the geodesic distance (in radians) between R1 and R2 < epsilon, False otherwise.
    """
    epsilon = 0.0001
    
    if abs(max(map(max, R1 - R2))) <= epsilon and abs(min(map(min, R1 - R2))) <= epsilon:
        return True 
    else:
        return False

def cosine_value_clip(c):
    threshold = 10**(-5)
    if abs(c)>1 and abs(c) <= 1 + threshold:
        c = np.sign(c)
        return c
    else:
        return c

def C_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type C.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    # cosine of angle of the path
    c_phi1 = (r_min**2 - Alpha[2, 2]) / (r_min**2 - 1)
    c_phi1 = cosine_value_clip(c_phi1) 
    if c_phi1 >= -1 and c_phi1 <= 1:  # if phi1 has solution
        phi1_sol1 = acos(c_phi1)
        phi1_sol2 = 2 * pi - phi1_sol1
        
        C_type_list = ["L+", "R+", "R-", "L-"]

        # traverse all four path types, add types and lengths to list
        for enum in C_type_list:
            type_list.append({"path": [enum], "angles": [phi1_sol1]})
            type_list.append({"path": [enum], "angles": [phi1_sol2]})
        return type_list
    else:
        return type_list


def G_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type G.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    # cosine of angle of the path
    c_phi1 = Alpha[0, 0] + r_min * (Alpha[0, 2] * sqrt(1 - r_min**2) - Alpha[2, 0] * sqrt(1 - r_min**2) + (Alpha[2, 2] - 1) * r_min) / (r_min**2 - 1)
    c_phi1 = cosine_value_clip(c_phi1)
    if c_phi1 >= -1 and c_phi1 <= 1:  # if phi1 has solution
        phi1_sol1 = acos(c_phi1)
        phi1_sol2 = 2 * pi - phi1_sol1
        C_type_list = ["G+", "G-"]

        # traverse both path types, add types and lengths to list
        for enum in C_type_list:
            type_list.append({"path": [enum], "angles": [phi1_sol1]})
            type_list.append({"path": [enum], "angles": [phi1_sol2]})
        return type_list
    else:
        return type_list
    

def T_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type T.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    # cosine of angle of the path
    c_phi1 = Alpha[2, 2]
    c_phi1 = cosine_value_clip(c_phi1)
    if c_phi1 >= -1 and c_phi1 <= 1:  # if phi1 has solution
        phi1_sol1 = acos(c_phi1)
        phi1_sol2 = 2 * pi - phi1_sol1
        C_type_list = ["L0", "R0"]

        # traverse both path types, add types and lengths to list
        for enum in C_type_list:
            type_list.append({"path": [enum], "angles": [phi1_sol1]})
            type_list.append({"path": [enum], "angles": [phi1_sol2]})
        return type_list
    else:
        return type_list


def CC_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type CC.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    # cosine of angle of the L+R+ and R-L- paths
    c_phi1 = (-2 * r_min**3 - Alpha[2, 0] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min + r_min) / (2 * r_min - 2 * r_min**3)
    c_phi1 = cosine_value_clip(c_phi1)

    c_phi2 = (-2 * r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min + r_min) / (2 * r_min - 2 * r_min**3)
    c_phi2 = cosine_value_clip(c_phi2)

    if (c_phi1 >= -1 and c_phi1 <= 1 and c_phi2 >= -1 and c_phi2 <= 1):  # if phi1 and phi2 have solution
        phi1_sol = [acos(c_phi1), 2 * pi - acos(c_phi1)]
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        # add candidate paths to list
        for enum1 in phi1_sol:
            for enum2 in phi2_sol:
                type_list.append({"path": ["L+", "R+"], "angles": [enum1, enum2]})
                type_list.append({"path": ["R-", "L-"], "angles": [enum1, enum2]})

    # cosine of angle of the R+L+ and L-R- paths
    c_phi1 = (-2 * r_min**3 + Alpha[2, 0] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min + r_min) / (2 * r_min - 2 * r_min**3)
    c_phi1 = cosine_value_clip(c_phi1)

    c_phi2 = (-2 * r_min**3 - Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min + r_min) / (2 * r_min - 2 * r_min**3)
    c_phi2 = cosine_value_clip(c_phi2)

    if ( c_phi1 >= -1 and c_phi1 <= 1 and c_phi2 >= -1 and c_phi2 <= 1):  # if phi1 and phi2 have solution
        phi1_sol = [acos(c_phi1), 2 * pi - acos(c_phi1)]
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        # add candidate paths to list
        for enum1 in phi1_sol:
            for enum2 in phi2_sol:
                type_list.append({"path": ["R+", "L+"], "angles": [enum1, enum2]})
                type_list.append({"path": ["L-", "R-"], "angles": [enum1, enum2]})
    return type_list

def GC_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type GC.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    # cosine of angle of the G+L+ path
    c_phi1 = Alpha[0, 0] + r_min * (-Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 0] * sqrt(1 - r_min**2) + (Alpha[2, 2] - 1) * r_min) / (r_min**2 - 1)

    c_phi2 = (r_min**2 - Alpha[2, 2]) / (r_min**2 - 1)
    
    c_phi1 = cosine_value_clip(c_phi1) 
    c_phi2 = cosine_value_clip(c_phi2)

    if (
        c_phi1 >= -1 and c_phi1 <= 1 and c_phi2 >= -1 and c_phi2 <= 1
    ):  # if phi1 and phi2 have solution
        phi1_sol = [acos(c_phi1), 2 * pi - acos(c_phi1)]
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]

        for enum1 in phi1_sol:
            for enum2 in phi2_sol:
                type_list.append({"path": ["G+", "L+"], "angles": [enum1, enum2]})

    # cosine of angle of the G+R+ path
    c_phi1 = Alpha[0, 0] + r_min * (Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 0] * sqrt(1 - r_min**2) - (Alpha[2, 2] - 1) * r_min) / (r_min**2 - 1)

    c_phi2 = (r_min**2 - Alpha[2, 2]) / (r_min**2 - 1)

    c_phi1 = cosine_value_clip(c_phi1) 
    c_phi2 = cosine_value_clip(c_phi2)

    if (
        c_phi1 >= -1 and c_phi1 <= 1 and c_phi2 >= -1 and c_phi2 <= 1
    ):  # if phi1 and phi2 have solution
        phi1_sol = [acos(c_phi1), 2 * pi - acos(c_phi1)]
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        # add candidate paths to list
        for enum1 in phi1_sol:
            for enum2 in phi2_sol:
                type_list.append({"path": ["G+", "R+"], "angles": [enum1, enum2]})

    # cosine of angle of the G-R- path
    c_phi1 = Alpha[0, 0] - r_min * (Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 0] * sqrt(1 - r_min**2) + (Alpha[2, 2] - 1) * r_min) / (r_min**2 - 1)

    c_phi2 = (r_min**2 - Alpha[2, 2]) / (r_min**2 - 1)

    c_phi1 = cosine_value_clip(c_phi1) 
    c_phi2 = cosine_value_clip(c_phi2)

    if (
        c_phi1 >= -1 and c_phi1 <= 1 and c_phi2 >= -1 and c_phi2 <= 1
    ):  # if phi1 and phi2 have solution
        phi1_sol = [acos(c_phi1), 2 * pi - acos(c_phi1)]
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        # add candidate paths to list
        for enum1 in phi1_sol:
            for enum2 in phi2_sol:
                type_list.append({"path": ["G-", "R-"], "angles": [enum1, enum2]})

    # cosine of angle of the G-L- path
    c_phi1 = Alpha[0, 0] + r_min * (Alpha[0, 2] * sqrt(1 - r_min**2) - Alpha[2, 0] * sqrt(1 - r_min**2) + (Alpha[2, 2] - 1) * r_min) / (r_min**2 - 1)

    c_phi2 = (r_min**2 - Alpha[2, 2]) / (r_min**2 - 1)

    c_phi1 = cosine_value_clip(c_phi1) 
    c_phi2 = cosine_value_clip(c_phi2)


    if (
        c_phi1 >= -1 and c_phi1 <= 1 and c_phi2 >= -1 and c_phi2 <= 1
    ):  # if phi1 and phi2 have solution
        phi1_sol = [acos(c_phi1), 2 * pi - acos(c_phi1)]
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        # add candidate paths to list
        for enum1 in phi1_sol:
            for enum2 in phi2_sol:
                type_list.append({"path": ["G-", "L-"], "angles": [enum1, enum2]})
    return type_list


def C_C_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type C|C.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    # cosine of angle of the L+|L- and R-|R+ paths
    c_phi1 = (-2 * r_min**3 - Alpha[2, 0] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min + r_min) / (2 * r_min - 2 * r_min**3)

    c_phi2 = (-2 * r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min + r_min) / (2 * r_min - 2 * r_min**3)

    c_phi1 = cosine_value_clip(c_phi1) 
    c_phi2 = cosine_value_clip(c_phi2)

    if (
        c_phi1 >= -1 and c_phi1 <= 1 and c_phi2 >= -1 and c_phi2 <= 1
    ):  # if phi1 and phi2 have solution
        phi1_sol = [acos(c_phi1), 2 * pi - acos(c_phi1)]
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]

        for enum1 in phi1_sol:
            for enum2 in phi2_sol:
                type_list.append({"path": ["L+", "L-"], "angles": [enum1, enum2]})
                type_list.append({"path": ["R-", "R+"], "angles": [enum1, enum2]})

    # cosine of angle of the R+R- and L-|L+ paths
    c_phi1 = (-2 * r_min**3 + Alpha[2, 0] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min + r_min) / (2 * r_min - 2 * r_min**3)

    c_phi2 = (-2 * r_min**3 - Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min + r_min) / (2 * r_min - 2 * r_min**3)

    c_phi1 = cosine_value_clip(c_phi1) 
    c_phi2 = cosine_value_clip(c_phi2)

    if (
        c_phi1 >= -1 and c_phi1 <= 1 and c_phi2 >= -1 and c_phi2 <= 1
    ):  # if phi1 and phi2 have solution
        phi1_sol = [acos(c_phi1), 2 * pi - acos(c_phi1)]
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        
        # add candidate paths to list
        for enum1 in phi1_sol:
            for enum2 in phi2_sol:
                type_list.append({"path": ["R+", "R-"], "angles": [enum1, enum2]})
                type_list.append({"path": ["L-", "L+"], "angles": [enum1, enum2]})
    return type_list


def TC_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type TC.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    # cosine of angle of the L0L+ and R0R- paths
    c_phi1 = Alpha[2,2] + (Alpha[2,0]*sqrt(1-r_min**2))/r_min

    c_phi2 = 1 - Alpha[0,2]/(r_min*sqrt(1-r_min**2))

    c_phi1 = cosine_value_clip(c_phi1) 
    c_phi2 = cosine_value_clip(c_phi2)

    if (
        c_phi1 >= -1 and c_phi1 <= 1 and c_phi2 >= -1 and c_phi2 <= 1
    ):  # if phi1 and phi2 have solution
        phi1_sol = [acos(c_phi1), 2 * pi - acos(c_phi1)]
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]

        for enum1 in phi1_sol:
            for enum2 in phi2_sol:
                type_list.append({"path": ["L0", "L+"], "angles": [enum1, enum2]})
                type_list.append({"path": ["R0", "R-"], "angles": [enum1, enum2]})

    # cosine of angle of the R0R+ and L0L- paths
    c_phi1 = Alpha[2,2] - (Alpha[2,0]*sqrt(1-r_min**2))/r_min

    c_phi2 = 1 + Alpha[0,2]/(r_min*sqrt(1-r_min**2))

    c_phi1 = cosine_value_clip(c_phi1) 
    c_phi2 = cosine_value_clip(c_phi2)

    if (
        c_phi1 >= -1 and c_phi1 <= 1 and c_phi2 >= -1 and c_phi2 <= 1
    ):  # if phi1 and phi2 have solution
        phi1_sol = [acos(c_phi1), 2 * pi - acos(c_phi1)]
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        
        # add candidate paths to list
        for enum1 in phi1_sol:
            for enum2 in phi2_sol:
                type_list.append({"path": ["L0", "L-"], "angles": [enum1, enum2]})
                type_list.append({"path": ["R0", "R+"], "angles": [enum1, enum2]})
    return type_list


def CC_C_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type CC|C.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    # cosine of second angle of the L+R+|R- and R-L-|L+ paths
    c_psi = (-Alpha[0, 0] + 4 * r_min**4 + Alpha[0, 0] * r_min**2 - Alpha[2, 2] * r_min**2 - Alpha[0, 2] * sqrt(1 - r_min**2) * r_min
             - Alpha[2, 0] * sqrt(1 - r_min**2) * r_min - 4 * r_min**2 + 1) / (4 * r_min**2 * (r_min**2 - 1))
    c_psi = cosine_value_clip(c_psi) 

    if c_psi >= -1 and c_psi <= 1:  # if psi has solution
        psi_sol = [acos(c_psi), 2 * pi - acos(c_psi)]
        U_max = sqrt(1/r_min**2 - 1)
        for psi in psi_sol:
            if psi <= atan(1/sqrt(U_max**4-1)) + pi/2 + 0.00001:
                # cos(phi1+gamma)
                c_phi1_plus_gamma = -(r_min * (Alpha[2, 2] + 4 * (r_min**2 - 1) * r_min**2 * cos(psi) - (1 - 2 * r_min**2)**2) + 
                                      Alpha[2, 0] * sqrt(1 - r_min**2)) / max((4* sqrt(-r_min**2 * (r_min**2 - 1)**2 * sin(psi / 2)**2* (-2 * r_min**4 + 
                                        2 * (r_min**2 - 1) * r_min**2 * cos(psi) + 2 * r_min**2 - 1))) , 10**(-8))
                # cos(phi2-gamma)
                c_phi2_minus_gamma = -(r_min * (Alpha[2, 2] + 4 * (r_min**2 - 1) * r_min**2 * cos(psi) - (1 - 2 * r_min**2)**2) + 
                                       Alpha[0, 2] * sqrt(1 - r_min**2)) / max((4 * sqrt(-r_min**2 * (r_min**2 - 1)**2 * sin(psi / 2)**2 * (-2 * r_min**4
                                       + 2 * (r_min**2 - 1) * r_min**2 * cos(psi) + 2 * r_min**2 - 1))) , 10**(-8))
                
                c_phi1_plus_gamma = cosine_value_clip(c_phi1_plus_gamma) 
                c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma) 
                
                if c_phi1_plus_gamma >= -1 and c_phi1_plus_gamma <= 1 and c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if have solution
                    phi1_plus_gamma_sol = [acos(c_phi1_plus_gamma), 2 * pi - acos(c_phi1_plus_gamma)]
                    phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
                    
                    for enum1 in phi1_plus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 - atan2((2 * r_min * (sin(psi) - r_min**2 * sin(psi))) , (4 * r_min * (2 * r_min**4 - 3 * r_min**2 + 1) * sin(psi / 2)**2))
                        
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        
                        for enum2 in phi2_minus_gamma_sol:
                            # calculate phi2
                            phi2 = enum2 + atan2((2 * r_min * (sin(psi) - r_min**2 * sin(psi))) , (4 * r_min * (2 * r_min**4 - 3 * r_min**2 + 1) * sin(psi / 2)**2))
                            
                            if abs(phi2-2*pi)<10**(-3):
                                phi2 = 0.0
                            if phi2 < -10**(-3):
                                phi2 += 2 * pi
                            elif phi2 > 2 * pi:
                                phi2 -= 2 * pi
                            
                            # add candidate paths to list
                            type_list.append({"path": ["L+", "R+", "R-"], "angles": [phi1, psi, phi2]})
                            type_list.append({"path": ["R-", "L-", "L+"], "angles": [phi1, psi, phi2]})
                        

    # cosine of second angle of the L-R-|R+ and R+L+|L- paths
    c_psi = (-Alpha[0, 0] + 4 * r_min**4 + Alpha[0, 0] * r_min**2 - Alpha[2, 2] * r_min**2 + Alpha[0, 2] * sqrt(1 - r_min**2) * 
             r_min + Alpha[2, 0] * sqrt(1 - r_min**2) * r_min - 4 * r_min**2 + 1) / (4 * r_min**2 * (r_min**2 - 1))
    
    c_psi = cosine_value_clip(c_psi) 
    

    if c_psi >= -1 and c_psi <= 1:  # if psi has solution
        psi_sol = [acos(c_psi), 2 * pi - acos(c_psi)]
        U_max = sqrt(1/r_min**2 - 1)
        for psi in psi_sol:
            if psi <= atan(1/sqrt(U_max**4-1)) + pi/2+ 0.00001: # upper bound on psi
                # cos(phi1+gamma)
                c_phi1_plus_gamma = -(r_min * ( Alpha[2, 2] + 4 * (r_min**2 - 1) * r_min**2 * cos(psi) - (1 - 2 * r_min**2)**2) - Alpha[2, 0] * 
                                      sqrt(1 - r_min**2)) / max((4 * sqrt(-r_min**2 * (r_min**2 - 1)**2 * sin(psi / 2)**2 * (-2 * r_min**4
                                      + 2 * (r_min**2 - 1) * r_min**2 * cos(psi) + 2 * r_min**2 - 1))) , 10**(-8))
                # cos(phi2-gamma)
                c_phi2_minus_gamma = -(r_min * (Alpha[2, 2] + 4 * (r_min**2 - 1) * r_min**2 * cos(psi) - (1 - 2 * r_min**2)**2) - Alpha[0, 2] * 
                                       sqrt(1 - r_min**2)) / max((4 * sqrt(-r_min**2 * (r_min**2 - 1)**2 * sin(psi / 2)**2 * (-2 * r_min**4
                                      + 2 * (r_min**2 - 1) * r_min**2 * cos(psi) + 2 * r_min**2 - 1))) , 10**(-8))
                
                c_phi1_plus_gamma = cosine_value_clip(c_phi1_plus_gamma) 
                c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma) 
                

                if c_phi1_plus_gamma >= -1 and c_phi1_plus_gamma <= 1 and c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if have solution
                    phi1_plus_gamma_sol = [acos(c_phi1_plus_gamma), 2 * pi - acos(c_phi1_plus_gamma)]
                    phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
                    
                    for enum1 in phi1_plus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 - atan2((2 * r_min * (sin(psi) - r_min**2 * sin(psi))) , (4 * r_min * (2 * r_min**4 - 3 * r_min**2 + 1) * sin(psi / 2)**2))
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        
                        for enum2 in phi2_minus_gamma_sol:
                            # calculate phi2
                            phi2 = enum2 + atan2((2 * r_min * (sin(psi) - r_min**2 * sin(psi))) , (4 * r_min * (2 * r_min**4 - 3 * r_min**2 + 1) * sin(psi / 2)**2))
                            if abs(phi2-2*pi)<10**(-3):
                                phi2 = 0.0
                            if phi2 < -10**(-3):
                                phi2 += 2 * pi
                            elif phi2 > 2 * pi:
                                phi2 -= 2 * pi
                            # add candidate paths to list
                            type_list.append({"path": ["L-", "R-", "R+"], "angles": [phi1, psi, phi2]})
                            type_list.append({"path": ["R+", "L+", "L-"], "angles": [phi1, psi, phi2]})
    return type_list


def CGC_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type CGC.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    # cosine of second angle of the L+G+L+ paths
    c_phi2 = (-Alpha[0, 0]+ Alpha[0, 0] * r_min**2- Alpha[2, 2] * r_min**2- Alpha[0, 2] * sqrt(1 - r_min**2) * r_min
             - Alpha[2, 0] * sqrt(1 - r_min**2) * r_min+ r_min**2) / (r_min**2 - 1)
    c_phi2 = cosine_value_clip(c_phi2)
   
    if c_phi2 >= -1 and c_phi2 <= 1:  # if phi2 has solution
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        for phi2 in phi2_sol:
                # cos(phi1-gamma)
                c_phi1_minus_gamma = (r_min * (-Alpha[2, 2] + r_min**2 * (-cos(phi2)) + r_min**2 + cos(phi2)) - Alpha[2, 0] 
                                     * sqrt(1 - r_min**2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * sin(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-15))
                # cos(phi2-gamma)
                c_phi3_minus_gamma = (r_min * (-Alpha[2, 2] + r_min**2 * (-cos(phi2)) + r_min**2 + cos(phi2)) - Alpha[0, 2] 
                                     * sqrt(1 - r_min**2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * sin(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-15))
                
                c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                c_phi3_minus_gamma = cosine_value_clip(c_phi3_minus_gamma) 

                if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi3_minus_gamma >= -1 and c_phi3_minus_gamma <= 1:  # if have solution
                    phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                    phi3_minus_gamma_sol = [acos(c_phi3_minus_gamma), 2 * pi - acos(c_phi3_minus_gamma)]
                    for enum1 in phi1_minus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 + atan2((r_min**2 * sin(phi2) - sin(phi2)) , (2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        for enum2 in phi3_minus_gamma_sol:
                            # calculate phi3
                            phi3 = enum2 + atan2((r_min**2 * sin(phi2) - sin(phi2)) , (2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                            if abs(phi3-2*pi)<10**(-3):
                                phi3 = 0.0
                            if phi3 < -10**(-3):
                                phi3 += 2 * pi
                            elif phi3 > 2 * pi:
                                phi3 -= 2 * pi

                            # add candidate path to list
                            type_list.append({"path": ["L+", "G+", "L+"], "angles": [phi1, phi2, phi3]})
                            
                        

    # cosine of second angle of the R+G+R+ paths
    c_phi2 = (-Alpha[0, 0]+ Alpha[0, 0] * r_min**2- Alpha[2, 2] * r_min**2 + Alpha[0, 2] * sqrt(1 - r_min**2) * r_min
             + Alpha[2, 0] * sqrt(1 - r_min**2) * r_min+ r_min**2) / (r_min**2 - 1)
    
    c_phi2 = cosine_value_clip(c_phi2)

    if c_phi2 >= -1 and c_phi2 <= 1:  # if phi2 has solution
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        
        for phi2 in phi2_sol:
                # cos(phi1-gamma)
                c_phi1_minus_gamma = (r_min * (-Alpha[2, 2] + r_min**2 * (-cos(phi2)) + r_min**2 + cos(phi2)) + Alpha[2, 0] 
                                     * sqrt(1 - r_min**2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * sin(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-8))
                # cos(phi2-gamma)
                c_phi3_minus_gamma = (r_min * (-Alpha[2, 2] + r_min**2 * (-cos(phi2)) + r_min**2 + cos(phi2)) + Alpha[0, 2] 
                                     * sqrt(1 - r_min**2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * sin(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-8))
                
                c_phi1_minus_gamma  = cosine_value_clip(c_phi1_minus_gamma)
                c_phi3_minus_gamma = cosine_value_clip(c_phi3_minus_gamma)
                
                

                if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi3_minus_gamma >= -1 and c_phi3_minus_gamma <= 1:  # if have solution
                    phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                    phi3_minus_gamma_sol = [acos(c_phi3_minus_gamma), 2 * pi - acos(c_phi3_minus_gamma)]
                    
                    for enum1 in phi1_minus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 + atan2((r_min**2 * sin(phi2) - sin(phi2)) , (2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        
                        for enum2 in phi3_minus_gamma_sol:
                            # calculate phi3
                            phi3 = enum2 + atan2((r_min**2 * sin(phi2) - sin(phi2)) , (2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                            if abs(phi3-2*pi)<10**(-3):
                                phi3 = 0.0
                            if phi3 < -10**(-3):
                                phi3 += 2 * pi
                            elif phi3 > 2 * pi:
                                phi3 -= 2 * pi
                            
                            # add candidate path to list
                            type_list.append({"path": ["R+", "G+", "R+"], "angles": [phi1, phi2, phi3]})
                            

    # cosine of second angle of the R-G-R- paths
    c_phi2 = (-Alpha[0, 0]+ Alpha[0, 0] * r_min**2- Alpha[2, 2] * r_min**2- Alpha[0, 2] * sqrt(1 - r_min**2) * r_min
             - Alpha[2, 0] * sqrt(1 - r_min**2) * r_min+ r_min**2) / (r_min**2 - 1)
    c_phi2 = cosine_value_clip(c_phi2)

    if c_phi2 >= -1 and c_phi2 <= 1:  # if phi2 has solution
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        
        for phi2 in phi2_sol:
                # cos(phi1-gamma)
                c_phi1_minus_gamma = (r_min * (-Alpha[2, 2] + r_min**2 * (-cos(phi2)) + r_min**2 + cos(phi2)) - Alpha[2, 0] 
                                     * sqrt(1 - r_min**2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * sin(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-8))
                # cos(phi2-gamma)
                c_phi3_minus_gamma = (r_min * (-Alpha[2, 2] + r_min**2 * (-cos(phi2)) + r_min**2 + cos(phi2)) - Alpha[0, 2] 
                                     * sqrt(1 - r_min**2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * sin(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-8))
                
                c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                c_phi3_minus_gamma = cosine_value_clip(c_phi3_minus_gamma)
                

                if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi3_minus_gamma >= -1 and c_phi3_minus_gamma <= 1:  # if have solution
                    phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                    phi3_minus_gamma_sol = [acos(c_phi3_minus_gamma), 2 * pi - acos(c_phi3_minus_gamma)]
                    
                    for enum1 in phi1_minus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 + atan2((r_min**2 * sin(phi2) - sin(phi2)) , (2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        
                        for enum2 in phi3_minus_gamma_sol:
                            # calculate phi3
                            phi3 = enum2 + atan2((r_min**2 * sin(phi2) - sin(phi2)) , (2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                            if abs(phi3-2*pi)<10**(-3):
                                phi3 = 0.0
                            if phi3 < -10**(-3):
                                phi3 += 2 * pi
                            elif phi3 > 2 * pi:
                                phi3 -= 2 * pi
                            
                            # add candidate path to list
                            type_list.append({"path": ["R-", "G-", "R-"], "angles": [phi1, phi2, phi3]})

    # cosine of second angle of the L-G-L- paths
    c_phi2 = (-Alpha[0, 0]+ Alpha[0, 0] * r_min**2- Alpha[2, 2] * r_min**2 + Alpha[0, 2] * sqrt(1 - r_min**2) * r_min
             + Alpha[2, 0] * sqrt(1 - r_min**2) * r_min+ r_min**2) / (r_min**2 - 1)
    c_phi2 = cosine_value_clip(c_phi2)

    if c_phi2 >= -1 and c_phi2 <= 1:  # if phi2 has solution
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        
        for phi2 in phi2_sol:
                # cos(phi1-gamma)
                c_phi1_minus_gamma = (r_min * (-Alpha[2, 2] + r_min**2 * (-cos(phi2)) + r_min**2 + cos(phi2)) + Alpha[2, 0] 
                                     * sqrt(1 - r_min**2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * sin(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-8))
                # cos(phi2-gamma)
                c_phi3_minus_gamma = (r_min * (-Alpha[2, 2] + r_min**2 * (-cos(phi2)) + r_min**2 + cos(phi2)) + Alpha[0, 2] 
                                     * sqrt(1 - r_min**2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * sin(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-8))
                
                c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                c_phi3_minus_gamma = cosine_value_clip(c_phi3_minus_gamma)

                if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi3_minus_gamma >= -1 and c_phi3_minus_gamma <= 1:  # if have solution
                    phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                    phi3_minus_gamma_sol = [acos(c_phi3_minus_gamma), 2 * pi - acos(c_phi3_minus_gamma)]
                    
                    for enum1 in phi1_minus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 + atan2((r_min**2 * sin(phi2) - sin(phi2)) , (2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        
                        for enum2 in phi3_minus_gamma_sol:
                            # calculate phi3
                            phi3 = enum2 + atan2((r_min**2 * sin(phi2) - sin(phi2)) , (2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                            if abs(phi3-2*pi)<10**(-3):
                                phi3 = 0.0
                            if phi3 < -10**(-3):
                                phi3 += 2 * pi
                            elif phi3 > 2 * pi:
                                phi3 -= 2 * pi
                            
                            # add candidate path to list
                            type_list.append({"path": ["L-", "G-", "L-"], "angles": [phi1, phi2, phi3]})

    # cosine of second angle of the L+G+R+ and R-G-L- paths
    c_phi2 = (-Alpha[0, 0]+ Alpha[0, 0] * r_min**2 + Alpha[2, 2] * r_min**2 + Alpha[0, 2] * sqrt(1 - r_min**2) * r_min
             - Alpha[2, 0] * sqrt(1 - r_min**2) * r_min - r_min**2) / (r_min**2 - 1)
    c_phi2 = cosine_value_clip(c_phi2)

    if c_phi2 >= -1 and c_phi2 <= 1:  # if phi2 has solution
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        
        for phi2 in phi2_sol:
                # cos(phi1-gamma)
                c_phi1_minus_gamma = (r_min**3 * cos(phi2) + r_min**3 + Alpha[2, 0] * sqrt(1 - r_min**2) - Alpha[2, 2] * r_min - r_min *
                                       cos(phi2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * cos(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-8))
                # cos(phi2-gamma)
                c_phi3_minus_gamma = (r_min**3 * cos(phi2) + r_min**3 - Alpha[0, 2] * sqrt(1 - r_min**2) - Alpha[2, 2] * r_min - r_min *
                                       cos(phi2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * cos(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-8))
                
                c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                c_phi3_minus_gamma = cosine_value_clip(c_phi3_minus_gamma)

                if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi3_minus_gamma >= -1 and c_phi3_minus_gamma <= 1:  # if have solution
                    phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                    phi3_minus_gamma_sol = [acos(c_phi3_minus_gamma), 2 * pi - acos(c_phi3_minus_gamma)]
                    
                    for enum1 in phi1_minus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 + atan2((sin(phi2) - r_min**2 * sin(phi2)) , (2 * (r_min**2 - 1) * r_min * cos(phi2 / 2)**2))
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        
                        for enum2 in phi3_minus_gamma_sol:
                            # calculate phi3
                            phi3 = enum2 + atan2((sin(phi2) - r_min**2 * sin(phi2)) , (2 * (r_min**2 - 1) * r_min * cos(phi2 / 2)**2))
                            if abs(phi3-2*pi)<10**(-3):
                                phi3 = 0.0
                            if phi3 < -10**(-3):
                                phi3 += 2 * pi
                            elif phi3 > 2 * pi:
                                phi3 -= 2 * pi
                            
                            # add candidate path to list
                            type_list.append({"path": ["L+", "G+", "R+"], "angles": [phi1, phi2, phi3]})
                            type_list.append({"path": ["R-", "G-", "L-"], "angles": [phi1, phi2, phi3]})


    # cosine of second angle of the R+G+L+ and L-G-R- paths
    c_phi2 = (-Alpha[0, 0]+ Alpha[0, 0] * r_min**2 + Alpha[2, 2] * r_min**2 - Alpha[0, 2] * sqrt(1 - r_min**2) * r_min
             + Alpha[2, 0] * sqrt(1 - r_min**2) * r_min - r_min**2) / (r_min**2 - 1)
    c_phi2 = cosine_value_clip(c_phi2)

    if c_phi2 >= -1 and c_phi2 <= 1:  # if phi2 has solution
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        
        for phi2 in phi2_sol:
                # cos(phi1-gamma)
                c_phi1_minus_gamma = (r_min**3 * cos(phi2) + r_min**3 - Alpha[2, 0] * sqrt(1 - r_min**2) - Alpha[2, 2] * r_min - r_min *
                                       cos(phi2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * cos(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-8))
                # cos(phi2-gamma)
                c_phi3_minus_gamma = (r_min**3 * cos(phi2) + r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) - Alpha[2, 2] * r_min - r_min *
                                       cos(phi2)) / max(sqrt((r_min**2 - 1)**2 * (4 * r_min**2 * cos(phi2 / 2)**4 + sin(phi2)**2)) , 10**(-8))
                
                c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                c_phi3_minus_gamma = cosine_value_clip(c_phi3_minus_gamma)

                if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi3_minus_gamma >= -1 and c_phi3_minus_gamma <= 1:  # if have solution
                    phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                    phi3_minus_gamma_sol = [acos(c_phi3_minus_gamma), 2 * pi - acos(c_phi3_minus_gamma)]
                    
                    for enum1 in phi1_minus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 + atan2((sin(phi2) - r_min**2 * sin(phi2)) , (2 * (r_min**2 - 1) * r_min * cos(phi2 / 2)**2))
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        
                        for enum2 in phi3_minus_gamma_sol:
                            # calculate phi3
                            phi3 = enum2 + atan2((sin(phi2) - r_min**2 * sin(phi2)) , (2 * (r_min**2 - 1) * r_min * cos(phi2 / 2)**2))
                            if abs(phi3-2*pi)<10**(-3):
                                phi3 = 0.0
                            if phi3 < -10**(-3):
                                phi3 += 2 * pi
                            elif phi3 > 2 * pi:
                                phi3 -= 2 * pi
                            
                            # add candidate path to list
                            type_list.append({"path": ["R+", "G+", "L+"], "angles": [phi1, phi2, phi3]})
                            type_list.append({"path": ["L-", "G-", "R-"], "angles": [phi1, phi2, phi3]})
    return type_list


def C_CG_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type C|CG.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    U_max = sqrt(1/r_min**2 - 1)
    # second angle of the L+|L-G- and R-|R+G+ paths, which is known
    beta = atan(1/sqrt(U_max**4-1)) + pi/2

    # cos(phi1-gamma)
    c_phi1_minus_gamma = (Alpha[2, 2] + r_min**2 * (2 * (r_min**2 - 1) * cos(beta) - 2 * r_min**2 + 1)) / max(sqrt((r_min**2 - 1)**2 * 
                          (6 * r_min**4 + 2 * (r_min**2 - 1) * r_min**2 * cos(2 * beta) - 2 * r_min**2 + (4 * r_min**2 - 8 * r_min**4) * cos(beta) + 1)) , 10**(-8))
    # cos(phi2-gamma)
    c_phi2_minus_theta = (r_min * (2 * r_min**3 * cos(beta) - 2 * r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 0] * 
                            sqrt(1 - r_min**2) + Alpha[2, 2] * r_min - 2 * r_min * cos(beta) + r_min) - Alpha[0, 0] * (r_min**2 - 1)) / max(sqrt((r_min**2 - 1)**2 * 
                            (4 * r_min**4 * cos(beta)**2 + 4 * r_min**2 * sin(beta)**2 + (1 - 2 * r_min**2)**2 + (4 * r_min**2 - 8 * r_min**4) * cos(beta))) , 10**(-8))
    
    c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
    c_phi2_minus_theta = cosine_value_clip(c_phi2_minus_theta)

    if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi2_minus_theta >= -1 and c_phi2_minus_theta <= 1:  # if have solution
        phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
        phi2_minus_theta_sol = [acos(c_phi2_minus_theta), 2 * pi - acos(c_phi2_minus_theta)]
        
        for enum1 in phi1_minus_gamma_sol:
            # calculate phi1
            phi1 = enum1 + atan2((r_min**2 * sin(beta) - sin(beta)) , (-2 * r_min**4 + 2 * r_min**2 + (2 * r_min**4 - 3 * r_min**2 + 1) * cos(beta)))
            if abs(phi1-2*pi)<10**(-3):
                phi1 = 0.0
            if phi1 < -10**(-3):
                phi1 += 2 * pi
            elif phi1 > 2 * pi:
                phi1 -= 2 * pi
            
            for enum2 in phi2_minus_theta_sol:
                # calculate phi2
                phi2 = enum2 + atan2((2 * (r_min**2 - 1) * r_min * sin(beta)) , (2 * r_min**4 - 2 * (r_min**2 - 1) * r_min**2 * cos(beta) - 3 * r_min**2 + 1))
                if abs(phi2-2*pi)<10**(-3):
                    phi2 = 0.0
                if phi2 < -10**(-3):
                    phi2 += 2 * pi
                elif phi2 > 2 * pi:
                    phi2 -= 2 * pi
                # add candidate paths to list
                type_list.append({"path": ["L+", "L-", "G-"], "angles": [phi1, beta, phi2]})
                type_list.append({"path": ["R-", "R+", "G+"], "angles": [phi1, beta, phi2]})
                
                        

    # second angle of the R+|R-G- and L-|L+G+ paths, which is known
    beta = atan(1/sqrt(U_max**4-1)) + pi/2
    # cos(phi1-gamma)
    c_phi1_minus_gamma = (Alpha[2, 2] + r_min**2 * (2 * (r_min**2 - 1) * cos(beta) - 2 * r_min**2 + 1)) / max(sqrt((r_min**2 - 1)**2 * 
                          (6 * r_min**4 + 2 * (r_min**2 - 1) * r_min**2 * cos(2 * beta) - 2 * r_min**2 + (4 * r_min**2 - 8 * r_min**4) * cos(beta) + 1)) , 10**(-8))
    # cos(phi2-gamma)
    c_phi2_minus_theta = (r_min * (2 * r_min**3 * cos(beta) - 2 * r_min**3 - Alpha[0, 2] * sqrt(1 - r_min**2) - Alpha[2, 0] * 
                            sqrt(1 - r_min**2) + Alpha[2, 2] * r_min - 2 * r_min * cos(beta) + r_min) - Alpha[0, 0] * (r_min**2 - 1)) / max(sqrt((r_min**2 - 1)**2 * 
                            (4 * r_min**4 * cos(beta)**2 + 4 * r_min**2 * sin(beta)**2 + (1 - 2 * r_min**2)**2 + (4 * r_min**2 - 8 * r_min**4) * cos(beta))) , 10**(-8))
    c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
    c_phi2_minus_theta = cosine_value_clip(c_phi2_minus_theta)

    if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi2_minus_theta >= -1 and c_phi2_minus_theta <= 1:  # if have solution
        phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
        phi2_minus_theta_sol = [acos(c_phi2_minus_theta), 2 * pi - acos(c_phi2_minus_theta)]
        
        for enum1 in phi1_minus_gamma_sol:
            # calculate phi1
            phi1 = enum1 + atan2((r_min**2 * sin(beta) - sin(beta)) , (-2 * r_min**4 + 2 * r_min**2 + (2 * r_min**4 - 3 * r_min**2 + 1) * cos(beta)))
            if abs(phi1-2*pi)<10**(-3):
                phi1 = 0.0
            if phi1 < -10**(-3):
                phi1 += 2 * pi
            elif phi1 > 2 * pi:
                phi1 -= 2 * pi
            
            for enum2 in phi2_minus_theta_sol:
                # calculate phi2
                phi2 = enum2 + atan2((2 * (r_min**2 - 1) * r_min * sin(beta)) , (2 * r_min**4 - 2 * (r_min**2 - 1) * r_min**2 * cos(beta) - 3 * r_min**2 + 1))
                if abs(phi2-2*pi)<10**(-3):
                    phi2 = 0.0
                if phi2 < -10**(-3):
                    phi2 += 2 * pi
                elif phi2 > 2 * pi:
                    phi2 -= 2 * pi
                # add candidate paths to list
                type_list.append({"path": ["R+", "R-", "G-"], "angles": [phi1, beta, phi2]})
                type_list.append({"path": ["L-", "L+", "G+"], "angles": [phi1, beta, phi2]})
    return type_list


def CTC_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type CTC.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    # second angle of the L+L0L- and R-R0R+ paths
    c_phi2 = (-Alpha[0, 0] + Alpha[0, 0] * r_min**2 + Alpha[2, 2] * r_min**2 + Alpha[0, 2] * sqrt(1 - r_min**2) * r_min 
              - Alpha[2, 0] * sqrt(1 - r_min**2) *r_min - r_min**2 + 1) / r_min**2
    c_phi2 = cosine_value_clip(c_phi2)
    
    if c_phi2 >= -1 and c_phi2 <= 1:  # if phi2 has solution
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        for phi2 in phi2_sol:
                # cos(phi1-gamma)
                c_phi1_minus_gamma = -(r_min * (-Alpha[2, 2] + r_min**2 * cos(phi2) + r_min**2 - 1) + Alpha[2, 0] 
                                      * sqrt(1 - r_min**2)) / (max(sqrt(2) * sqrt(r_min**2 * (r_min**2 - 1) * cos(phi2 / 2)**2 * (r_min**2 * cos(phi2) + r_min**2 - 2)),10**(-15)))
                # cos(phi2-gamma)
                c_phi3_minus_gamma = (r_min**3 * (-cos(phi2)) - r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min + r_min) / (max(sqrt(2) * sqrt(r_min**2 
                                        * (r_min**2 - 1) * cos(phi2 / 2)**2 * (r_min**2 * cos(phi2) + r_min**2 - 2)),10**(-15)))

                c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                c_phi3_minus_gamma = cosine_value_clip(c_phi3_minus_gamma)

                if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi3_minus_gamma >= -1 and c_phi3_minus_gamma <= 1:  # if have solution
                    phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                    phi3_minus_gamma_sol = [acos(c_phi3_minus_gamma), 2 * pi - acos(c_phi3_minus_gamma)]
                    for enum1 in phi1_minus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 + atan2((-r_min*sqrt(1-r_min**2)*sin(phi2)) , (-2 * (r_min**2 - 1) * r_min * cos(phi2 / 2)**2))
                        #print(phi1)
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        #print(phi1)
                        for enum2 in phi3_minus_gamma_sol:
                            # calculate phi3
                            phi3 = enum2 + atan2((-r_min*sqrt(1-r_min**2)*sin(phi2)) , (-2 * (r_min**2 - 1) * r_min * cos(phi2 / 2)**2))
                            #print(phi3)
                            if abs(phi3-2*pi)<10**(-3):
                                phi3 = 0.0
                            if phi3 < -10**(-3):
                                phi3 += 2 * pi
                            elif phi3 > 2 * pi:
                                phi3 -= 2 * pi
                            #print(phi3)
                            
                            # add candidate path to list
                            type_list.append({"path": ["L+", "L0", "L-"], "angles": [phi1, phi2, phi3]})
                            type_list.append({"path": ["R-", "R0", "R+"], "angles": [phi1, phi2, phi3]})
                            
                        
    # second angle of the L-L0L+ and R+R0R- paths
    c_phi2 = (-Alpha[0, 0] + Alpha[0, 0] * r_min**2 + Alpha[2, 2] * r_min**2 - Alpha[0, 2] * sqrt(1 - r_min**2) * r_min 
              + Alpha[2, 0] * sqrt(1 - r_min**2) *r_min - r_min**2 + 1) / r_min**2
    c_phi2 = cosine_value_clip(c_phi2)
    
    if c_phi2 >= -1 and c_phi2 <= 1:  # if phi2 has solution
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        for phi2 in phi2_sol:
                # cos(phi1-gamma)
                c_phi1_minus_gamma = (r_min**3 * (-cos(phi2)) - r_min**3 + Alpha[2, 0] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min + r_min) / (max(sqrt(2) * sqrt(r_min**2 
                                        * (r_min**2 - 1) * cos(phi2 / 2)**2 * (r_min**2 * cos(phi2) + r_min**2 - 2)),10**(-15)))
                # cos(phi2-gamma)
                c_phi3_minus_gamma = -(r_min * (-Alpha[2, 2] + r_min**2 * cos(phi2) + r_min**2 - 1) + Alpha[0, 2] * sqrt(1 - r_min**2)) / (max(sqrt(2) 
                                        * sqrt(r_min**2 * (r_min**2 - 1) * cos(phi2 / 2)**2 * (r_min**2 * cos(phi2) + r_min**2 - 2)), 10**(-15)))
               
                c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                c_phi3_minus_gamma = cosine_value_clip(c_phi3_minus_gamma)
                
                if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi3_minus_gamma >= -1 and c_phi3_minus_gamma <= 1:  # if have solution
                    phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                    phi3_minus_gamma_sol = [acos(c_phi3_minus_gamma), 2 * pi - acos(c_phi3_minus_gamma)]
              
                    for enum1 in phi1_minus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 + atan2((-r_min*sqrt(1-r_min**2)*sin(phi2)) , (-2 * (r_min**2 - 1) * r_min * cos(phi2 / 2)**2))
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        for enum2 in phi3_minus_gamma_sol:
                            # calculate phi3
                            phi3 = enum2 + atan2((-r_min*sqrt(1-r_min**2)*sin(phi2)) , (-2 * (r_min**2 - 1) * r_min * cos(phi2 / 2)**2))
                            if abs(phi3-2*pi)<10**(-3):
                                phi3 = 0.0
                            if phi3 < -10**(-3):
                                phi3 += 2 * pi
                            elif phi3 > 2 * pi:
                                phi3 -= 2 * pi
                            # add candidate path to list
                            type_list.append({"path": ["R+", "R0", "R-"], "angles": [phi1, phi2, phi3]})
                            type_list.append({"path": ["L-", "L0", "L+"], "angles": [phi1, phi2, phi3]})
    

    # second angle of the L+L0L+ and R-R0R- paths
    c_phi2 = (Alpha[0, 0] - Alpha[0, 0] * r_min**2 + Alpha[2, 2] * r_min**2 + Alpha[0, 2] * sqrt(1 - r_min**2) 
              * r_min + Alpha[2, 0] * sqrt(1 - r_min**2) * r_min + r_min**2 - 1) / r_min**2
    c_phi2 = cosine_value_clip(c_phi2)
 
    if c_phi2 >= -1 and c_phi2 <= 1:  # if phi2 has solution
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        for phi2 in phi2_sol:
                # cos(phi1-gamma)
                c_phi1_minus_gamma = (r_min**3 * cos(phi2) - r_min**3 - Alpha[2, 0] * sqrt(1 - r_min**2) - Alpha[2, 2] * r_min + r_min) / (max(sqrt(2) * sqrt(-r_min**2 
                                        * (r_min**2 - 1) * sin(phi2 / 2)**2 * (r_min**2 * cos(phi2) - r_min**2 + 2)),10**(-15)))
                # cos(phi2-gamma)
                c_phi3_minus_gamma = (r_min**3 * cos(phi2) - r_min**3 - Alpha[0, 2] * sqrt(1 - r_min**2) - Alpha[2, 2] * r_min + r_min) / (max(sqrt(2) * sqrt(-r_min**2
                                        * (r_min**2 - 1) * sin(phi2 / 2)**2 * (r_min**2 * cos(phi2) - r_min**2 + 2)),10**(-15)))
                
                c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                c_phi3_minus_gamma = cosine_value_clip(c_phi3_minus_gamma)
                
                if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi3_minus_gamma >= -1 and c_phi3_minus_gamma <= 1:  # if have solution
                    phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                    phi3_minus_gamma_sol = [acos(c_phi3_minus_gamma), 2 * pi - acos(c_phi3_minus_gamma)]
                
                    for enum1 in phi1_minus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 + atan2((r_min*sqrt(1-r_min**2)*sin(phi2)) , (-2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                        if abs(phi1-2*pi)<10**(-3):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        for enum2 in phi3_minus_gamma_sol:
                            # calculate phi3
                            phi3 = enum2 + atan2((r_min*sqrt(1-r_min**2)*sin(phi2)) , (-2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                            if abs(phi3-2*pi)<10**(-3):
                                phi3 = 0.0
                            if phi3 < -10**(-3):
                                phi3 += 2 * pi
                            elif phi3 > 2 * pi:
                                phi3 -= 2 * pi
                            
                            # add candidate path to list
                            type_list.append({"path": ["L+", "L0", "L+"], "angles": [phi1, phi2, phi3]})
                            type_list.append({"path": ["R-", "R0", "R-"], "angles": [phi1, phi2, phi3]})
                            

    # second angle of the L-L0L- and R+R0R+ paths
    c_phi2 = (Alpha[0, 0] - Alpha[0, 0] * r_min**2 + Alpha[2, 2] * r_min**2 - Alpha[0, 2] * sqrt(1 - r_min**2) 
              * r_min - Alpha[2, 0] * sqrt(1 - r_min**2) * r_min + r_min**2 - 1) / r_min**2
    c_phi2 = cosine_value_clip(c_phi2)
    if c_phi2 >= -1 and c_phi2 <= 1:  # if phi2 has solution
        phi2_sol = [acos(c_phi2), 2 * pi - acos(c_phi2)]
        for phi2 in phi2_sol:
                # cos(phi1-gamma)
                c_phi1_minus_gamma = (r_min**3 * cos(phi2) - r_min**3 + Alpha[2, 0] * sqrt(1 - r_min**2) - Alpha[2, 2] * r_min + r_min) / (max(sqrt(2) * sqrt(-r_min**2 
                                        * (r_min**2 - 1) * sin(phi2 / 2)**2 * (r_min**2 * cos(phi2) - r_min**2 + 2)),10**(-15)))
                # cos(phi2-gamma)
                c_phi3_minus_gamma = (r_min**3 * cos(phi2) - r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) - Alpha[2, 2] * r_min + r_min) / (max(sqrt(2) * sqrt(-r_min**2
                                        * (r_min**2 - 1) * sin(phi2 / 2)**2 * (r_min**2 * cos(phi2) - r_min**2 + 2)),10**(-15)))
                
                c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                c_phi3_minus_gamma = cosine_value_clip(c_phi3_minus_gamma)
                
                if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi3_minus_gamma >= -1 and c_phi3_minus_gamma <= 1:  # if have solution
                    phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                    phi3_minus_gamma_sol = [acos(c_phi3_minus_gamma), 2 * pi - acos(c_phi3_minus_gamma)]
                    
                    for enum1 in phi1_minus_gamma_sol:
                        # calculate phi1
                        phi1 = enum1 + atan2((r_min*sqrt(1-r_min**2)*sin(phi2)) , (-2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                        if abs(phi1-2*pi)<10**(-4):
                            phi1 = 0.0
                        if phi1 < -10**(-3):
                            phi1 += 2 * pi
                        elif phi1 > 2 * pi:
                            phi1 -= 2 * pi
                        for enum2 in phi3_minus_gamma_sol:
                            # calculate phi3
                            phi3 = enum2 + atan2((r_min*sqrt(1-r_min**2)*sin(phi2)) , (-2 * (r_min**2 - 1) * r_min * sin(phi2 / 2)**2))
                            if abs(phi3-2*pi)<10**(-4):
                                phi3 = 0.0
                            if phi3 < -10**(-3):
                                phi3 += 2 * pi
                            elif phi3 > 2 * pi:
                                phi3 -= 2 * pi
                            
                            # add candidate path to list
                            type_list.append({"path": ["L-", "L0", "L-"], "angles": [phi1, phi2, phi3]})
                            type_list.append({"path": ["R+", "R0", "R+"], "angles": [phi1, phi2, phi3]})
                            
    return type_list


def C_CC_C_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type C|CC|C.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    U_max = sqrt(1/r_min**2 - 1)
    # cosine of second angle of the L+|L-R-|R+ and R-|R+L+|L- paths
    c_psi_sol1 = (4 * r_min**6 - 6 * r_min**4 + 2 * r_min**2 + sqrt(2) * sqrt(r_min**4 * (r_min**2 - 1) * (Alpha[2, 2] * r_min**2 + Alpha[0, 2] * sqrt(1 - r_min**2) * 
                    r_min - Alpha[2, 0] * sqrt(1 - r_min**2) * r_min + Alpha[0, 0] * (r_min**2 - 1) - 1))) / (4 * r_min**4 * (r_min**2 - 1))
    c_psi_sol2 = (4 * r_min**6 - 6 * r_min**4 + 2 * r_min**2 - sqrt(2) * sqrt(r_min**4 * (r_min**2 - 1) * (Alpha[2, 2] * r_min**2 + Alpha[0, 2] * sqrt(1 - r_min**2) * 
                    r_min - Alpha[2, 0] * sqrt(1 - r_min**2) * r_min + Alpha[0, 0] * (r_min**2 - 1) - 1))) / (4 * r_min**4 * (r_min**2 - 1))
    c_psi_sols=[c_psi_sol1, c_psi_sol2]
    for c_psi in c_psi_sols:
        c_psi = cosine_value_clip(c_psi)
        if c_psi >= -1 and c_psi <= 1:  # if psi has solution
            psi_sol = [acos(c_psi), 2 * pi - acos(c_psi)]
            for psi in psi_sol:
                if psi <= atan(1/sqrt(U_max**4-1)) + pi/2+ 0.00001:
                    A = 2 * (r_min**2 - 1) * r_min * sin(psi) * (2 * r_min**2 * cos(psi) - 2 * r_min**2 + 1)
                    B = -2 * (r_min**2 - 1) * r_min * ((8 * r_min**4 - 8 * r_min**2 + 1) * cos(psi) - (2 * r_min**2 - 1) * (r_min**2 * cos(2 * psi) + 3 * r_min**2 - 2))
                    # cos(phi1-gamma)
                    c_phi1_minus_gamma = (Alpha[2, 0] * sqrt(1 - r_min**2) - r_min * (Alpha[2, 2] - 4 * r_min**6 * cos(2 * psi) - 12 * r_min**6 + 4 * r_min**4 * 
                                            cos(2 * psi) + 20 * r_min**4 - 10 * r_min**2 + 8 * (2 * r_min**6 - 3 * r_min**4 + r_min**2) * cos(psi) + 1)) / max(sqrt(A**2 + B**2) , 10**(-8))
                    # cos(phi2-gamma)
                    c_phi2_minus_gamma = (Alpha[0, 2] * (-sqrt(1 - r_min**2)) - r_min * (Alpha[2, 2] - (4 * r_min**6 - 4 * r_min**4) * cos(2 * psi) - 12 *
                                            r_min**6 + 20 * r_min**4 - 10 * r_min**2 + 8 * (2 * r_min**6 - 3 * r_min**4 + r_min**2) * cos(psi) + 1)) / max(sqrt(A**2 + B**2) , 10**(-8))
                   
                    c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)

                    if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if have solution
                        phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
                        for enum1 in phi1_minus_gamma_sol:
                            # calculate phi1
                            phi1 = enum1 + atan2(A , B)
                            if abs(phi1-2*pi)<10**(-3):
                                phi1 = 0.0
                            if phi1 < -10**(-3):
                                phi1 += 2 * pi
                            elif phi1 > 2 * pi:
                                phi1 -= 2 * pi
                            for enum2 in phi2_minus_gamma_sol:
                                # calculate phi2
                                phi2 = enum2 + atan2(A , B)
                                if abs(phi2-2*pi)<10**(-3):
                                    phi2 = 0.0
                                if phi2 < -10**(-3):
                                    phi2 += 2 * pi
                                elif phi2 > 2 * pi:
                                    phi2 -= 2 * pi
                                # add candidate paths to list
                                type_list.append({"path": ["L+", "L-", "R-", "R+"], "angles": [phi1, psi, psi, phi2]})
                                type_list.append({"path": ["R-", "R+", "L+", "L-"], "angles": [phi1, psi, psi, phi2]})

    # cosine of second angle of the R+|R-L-|L+ and L-|L+R+|R- paths
    c_psi_sol1 = (4 * r_min**6 - 6 * r_min**4 + 2 * r_min**2 + sqrt(2) * sqrt(r_min**4 * (r_min**2 - 1) * (Alpha[2, 2] * r_min**2 - Alpha[0, 2] * sqrt(1 - r_min**2) * 
                    r_min + Alpha[2, 0] * sqrt(1 - r_min**2) * r_min + Alpha[0, 0] * (r_min**2 - 1) - 1))) / (4 * r_min**4 * (r_min**2 - 1))
    c_psi_sol2 = (4 * r_min**6 - 6 * r_min**4 + 2 * r_min**2 - sqrt(2) * sqrt(r_min**4 * (r_min**2 - 1) * (Alpha[2, 2] * r_min**2 - Alpha[0, 2] * sqrt(1 - r_min**2) * 
                    r_min + Alpha[2, 0] * sqrt(1 - r_min**2) * r_min + Alpha[0, 0] * (r_min**2 - 1) - 1))) / (4 * r_min**4 * (r_min**2 - 1))
    c_psi_sols=[c_psi_sol1, c_psi_sol2]
    for c_psi in c_psi_sols:
        c_psi = cosine_value_clip(c_psi)
        if c_psi >= -1 and c_psi <= 1:  # if psi has solution
            psi_sol = [acos(c_psi), 2 * pi - acos(c_psi)]
            for psi in psi_sol:
                if psi <= atan(1/sqrt(U_max**4-1)) + pi/2+ 0.00001:
                    A = 2 * (r_min**2 - 1) * r_min * sin(psi) * (2 * r_min**2 * cos(psi) - 2 * r_min**2 + 1)
                    B = -2 * (r_min**2 - 1) * r_min * ((8 * r_min**4 - 8 * r_min**2 + 1) * cos(psi) - (2 * r_min**2 - 1) * (r_min**2 * cos(2 * psi) + 3 * r_min**2 - 2))
                    # cos(phi1-gamma)
                    c_phi1_minus_gamma = (Alpha[2, 0] * (-sqrt(1 - r_min**2)) - r_min * (Alpha[2, 2] - (4 * r_min**6 - 4 * r_min**4) * cos(2 * psi) - 12 * r_min**6 + 
                                            20 * r_min**4 - 10 * r_min**2 + 8 * (2 * r_min**6 - 3 * r_min**4 + r_min**2) * cos(psi) + 1)) / max(sqrt(A**2 + B**2) , 10**(-8))
                    # cos(phi2-gamma)
                    c_phi2_minus_gamma = (Alpha[0, 2] * sqrt(1 - r_min**2) - r_min * (Alpha[2, 2] - (4 * r_min**6 - 4 * r_min**4) * cos(2 * psi) - 12 * r_min**6 + 
                                            20 * r_min**4 - 10 * r_min**2 + 8 * (2 * r_min**6 - 3 * r_min**4 + r_min**2) * cos(psi) + 1)) / max(sqrt(A**2 + B**2) , 10**(-8))
                    
                    c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)
                    

                    if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if have solution
                        phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
                        
                        for enum1 in phi1_minus_gamma_sol:
                            # calculate phi1
                            phi1 = enum1 + atan2(A , B)
                            if abs(phi1-2*pi)<10**(-3):
                                phi1 = 0.0
                            if phi1 < -10**(-3):
                                phi1 += 2 * pi
                            elif phi1 > 2 * pi:
                                phi1 -= 2 * pi
                            
                            for enum2 in phi2_minus_gamma_sol:
                                # calculate phi2
                                phi2 = enum2 + atan2(A , B)
                                if abs(phi2-2*pi)<10**(-3):
                                    phi2 = 0.0
                                if phi2 < -10**(-3):
                                    phi2 += 2 * pi
                                elif phi2 > 2 * pi:
                                    phi2 -= 2 * pi
                                # add candidate paths to list
                                type_list.append({"path": ["R+", "R-", "L-", "L+"], "angles": [phi1, psi, psi, phi2]})
                                type_list.append({"path": ["L-", "L+", "R+", "R-"], "angles": [phi1, psi, psi, phi2]})
    return type_list


def CGC_C_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type CGC|C.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    U_max = sqrt(1/r_min**2 - 1)
    # third angle of the L+G+L+|L- and R-G-R-|R+ paths, which is known
    beta = atan(1/sqrt(U_max**4-1)) + pi/2

    A = 2 * (r_min**2 - 1) * r_min * sin(beta)
    B = 2 * r_min**4 - 2 * (r_min**2 - 1) * r_min**2 * cos(beta) - 3 * r_min**2 + 1
    # cos(phi2-gamma)
    c_phi2_minus_gamma = (-(Alpha[0, 0] * (r_min**2 - 1)) - r_min * (2 * r_min**3 * cos(beta) - 2 * r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) - 
                            Alpha[2, 0] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min - 2 * r_min * cos(beta) + r_min)) / max(sqrt(A**2 + B**2) , 10**(-8))
    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)
    
    if c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if has solution
        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
        for enum2 in phi2_minus_gamma_sol:
            # calculate phi2
            phi2 = enum2 + atan2(A , B)
            if abs(phi2-2*pi)<10**(-3):
                phi2 = 0.0
            if phi2 < -10**(-3):
                phi2 += 2 * pi
            elif phi2 > 2 * pi:
                phi2 -= 2 * pi
        
            C = (r_min**2 - 1) * (sin(phi2) * (2 * r_min**2 * cos(beta) - 2 * r_min**2 + 1) + 2 * r_min * sin(beta) * cos(phi2))
            D = -(r_min**2 - 1) * r_min * (2 * cos(beta) * (r_min**2 * cos(phi2) - r_min**2 + 1) - 2 * r_min**2 * cos(phi2) + 2 * r_min**2 - 
                2 * r_min * sin(beta) * sin(phi2) + cos(phi2) - 1)
            # cos(phi1-theta)
            c_phi1_minus_theta = (r_min * (Alpha[2, 2] + (2 * r_min**4 - 3 * r_min**2 + 1) * cos(phi2) + r_min * (-2 * r_min**3 + 2 * (r_min**2 - 1) * sin(beta) * 
                                    sin(phi2) + 4 * (r_min**2 - 1) * r_min * cos(beta) * sin(phi2 / 2)**2 + r_min)) - Alpha[2, 0] * sqrt(1 - r_min**2)) / max(sqrt(C**2 + D**2) , 10**(-8))
            
            c_phi1_minus_theta = cosine_value_clip(c_phi1_minus_theta)
            
            E = -(r_min**2 - 1) * (cos(beta) * sin(phi2) + r_min * sin(beta) * (cos(phi2) - 1))
            F = -(r_min**2 - 1) * (sin(beta) * sin(phi2) + 2 * r_min**3 - 2 * r_min**2 * sin(beta) * sin(phi2) - 2 * (2 * r_min**2 - 1) * r_min * cos(beta) * 
                    sin(phi2 / 2)**2 - 2 * (r_min**2 - 1) * r_min * cos(phi2))
            # cos(phi3-gamma)
            c_phi3_minus_sigma = (r_min * (Alpha[2, 2] + (2 * r_min**4 - 3 * r_min**2 + 1) * cos(phi2) + r_min * (-2 * r_min**3 + 2 * (r_min**2 - 1) * 
                                    sin(beta) * sin(phi2) + 4 * (r_min**2 - 1) * r_min * cos(beta) * sin(phi2 / 2)**2 + r_min)) + Alpha[0, 2] * sqrt(1 - r_min**2)) / max(sqrt(E**2 + F**2) , 10**(-8))
            
            c_phi3_minus_sigma = cosine_value_clip(c_phi3_minus_sigma)

            if c_phi1_minus_theta >= -1 and c_phi1_minus_theta <= 1 and c_phi3_minus_sigma >= -1 and c_phi3_minus_sigma <= 1:  # if have solution
                phi1_minus_theta_sol = [acos(c_phi1_minus_theta), 2 * pi - acos(c_phi1_minus_theta)]
                phi3_minus_sigma_sol = [acos(c_phi3_minus_sigma), 2 * pi - acos(c_phi3_minus_sigma)]
                
                for enum1 in phi1_minus_theta_sol:
                    # calculate phi1
                    phi1 = enum1 + atan2(C , D)
                    if abs(phi1-2*pi)<10**(-3):
                        phi1 = 0.0
                    if phi1 < -10**(-3):
                        phi1 += 2 * pi
                    elif phi1 > 2 * pi:
                        phi1 -= 2 * pi
                    
                    for enum2 in phi3_minus_sigma_sol:
                        # calculate phi2
                        phi3 = enum2 + atan2(E , F)
                        if abs(phi3-2*pi)<10**(-3):
                            phi3 = 0.0
                        if phi3 < -10**(-3):
                            phi3 += 2 * pi
                        elif phi3 > 2 * pi:
                            phi3 -= 2 * pi
                        # add candidate paths to list
                        type_list.append({"path": ["L+", "G+", "L+", "L-"], "angles": [phi1, phi2, beta, phi3]})
                        type_list.append({"path": ["R-", "G-", "R-", "R+"], "angles": [phi1, phi2, beta, phi3]})

    # third angle of the R+G+R+|R- and  L-G-L-|L+ paths, which is known
    beta = atan(1/sqrt(U_max**4-1)) + pi/2

    A = 2 * (r_min**2 - 1) * r_min * sin(beta)
    B = 2 * r_min**4 - 2 * (r_min**2 - 1) * r_min**2 * cos(beta) - 3 * r_min**2 + 1
    # cos(phi2-gamma)
    c_phi2_minus_gamma = (-(Alpha[0, 0] * (r_min**2 - 1)) - r_min * (2 * r_min**3 * cos(beta) - 2 * r_min**3 - Alpha[0, 2] * sqrt(1 - r_min**2) + 
                            Alpha[2, 0] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min - 2 * r_min * cos(beta) + r_min)) / max(sqrt(A**2 + B**2) , 10**(-8))
    
    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)
    
    if c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if has solution
        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
        for enum2 in phi2_minus_gamma_sol:
            # calculate phi2
            phi2 = enum2 + atan2(A , B)
            if abs(phi2-2*pi)<10**(-3):
                phi2 = 0.0
            if phi2 < -10**(-3):
                phi2 += 2 * pi
            elif phi2 > 2 * pi:
                phi2 -= 2 * pi
        
            C = (r_min**2 - 1) * (sin(phi2) * (2 * r_min**2 * cos(beta) - 2 * r_min**2 + 1) + 2 * r_min * sin(beta) * cos(phi2))
            D = -(r_min**2 - 1) * r_min * (2 * cos(beta) * (r_min**2 * cos(phi2) - r_min**2 + 1) - 2 * r_min**2 * cos(phi2) + 2 * r_min**2 - 
                2 * r_min * sin(beta) * sin(phi2) + cos(phi2) - 1)
            # cos(phi1-theta)
            c_phi1_minus_theta = (r_min * (Alpha[2, 2] + (2 * r_min**4 - 3 * r_min**2 + 1) * cos(phi2) + r_min * (-2 * r_min**3 + 2 * (r_min**2 - 1) * sin(beta) * 
                                    sin(phi2) + 4 * (r_min**2 - 1) * r_min * cos(beta) * sin(phi2 / 2)**2 + r_min)) + Alpha[2, 0] * sqrt(1 - r_min**2)) / max(sqrt(C**2 + D**2) , 10**(-8))
            
            c_phi1_minus_theta = cosine_value_clip(c_phi1_minus_theta)
            
            E = -(r_min**2 - 1) * (cos(beta) * sin(phi2) + r_min * sin(beta) * (cos(phi2) - 1))
            F = -(r_min**2 - 1) * (sin(beta) * sin(phi2) + 2 * r_min**3 - 2 * r_min**2 * sin(beta) * sin(phi2) - 2 * (2 * r_min**2 - 1) * r_min * cos(beta) * 
                    sin(phi2 / 2)**2 - 2 * (r_min**2 - 1) * r_min * cos(phi2))
            # cos(phi3-gamma)
            c_phi3_minus_sigma = (r_min * (Alpha[2, 2] + (2 * r_min**4 - 3 * r_min**2 + 1) * cos(phi2) + r_min * (-2 * r_min**3 + 2 * (r_min**2 - 1) * 
                                    sin(beta) * sin(phi2) + 4 * (r_min**2 - 1) * r_min * cos(beta) * sin(phi2 / 2)**2 + r_min)) - Alpha[0, 2] * sqrt(1 - r_min**2)) / max(sqrt(E**2 + F**2) , 10**(-8))
            
            c_phi3_minus_sigma = cosine_value_clip(c_phi3_minus_sigma)
            

            if c_phi1_minus_theta >= -1 and c_phi1_minus_theta <= 1 and c_phi3_minus_sigma >= -1 and c_phi3_minus_sigma <= 1:  # if have solution
                phi1_minus_theta_sol = [acos(c_phi1_minus_theta), 2 * pi - acos(c_phi1_minus_theta)]
                phi3_minus_sigma_sol = [acos(c_phi3_minus_sigma), 2 * pi - acos(c_phi3_minus_sigma)]
                
                for enum1 in phi1_minus_theta_sol:
                    # calculate phi1
                    phi1 = enum1 + atan2(C , D)
                    if abs(phi1-2*pi)<10**(-3):
                        phi1 = 0.0
                    if phi1 < -10**(-3):
                        phi1 += 2 * pi
                    elif phi1 > 2 * pi:
                        phi1 -= 2 * pi
                    
                    for enum2 in phi3_minus_sigma_sol:
                        # calculate phi2
                        phi3 = enum2 + atan2(E , F)
                        if abs(phi3-2*pi)<10**(-3):
                            phi3 = 0.0
                        if phi3 < -10**(-3):
                            phi3 += 2 * pi
                        elif phi3 > 2 * pi:
                            phi3 -= 2 * pi
                        # add candidate paths to list
                        type_list.append({"path": ["R+", "G+", "R+", "R-"], "angles": [phi1, phi2, beta, phi3]})
                        type_list.append({"path": ["L-", "G-", "L-", "L+"], "angles": [phi1, phi2, beta, phi3]})

    # third angle of the L+G+R+|R- and R-G-L-|L+ paths, which is known
    beta = atan(1/sqrt(U_max**4-1)) + pi/2

    A = -2 * (r_min**2 - 1) * r_min * sin(beta)
    B = -(2 * r_min**4 - 2 * (r_min**2 - 1) * r_min**2 * cos(beta) - 3 * r_min**2 + 1)
    # cos(phi2-gamma)
    c_phi2_minus_gamma = (Alpha[0, 0] * (r_min**2 - 1) - r_min * (2 * r_min**3 * cos(beta) - 2 * r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 0] * 
                            sqrt(1 - r_min**2) + Alpha[2, 2] * r_min - 2 * r_min * cos(beta) + r_min)) / max(sqrt(A**2 + B**2) , 10**(-8))
    
    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)
    
    if c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if has solution
        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
        for enum2 in phi2_minus_gamma_sol:
            # calculate phi2
            phi2 = enum2 + atan2(A , B)
            if abs(phi2-2*pi)<10**(-3):
                phi2 = 0.0
            if phi2 < -10**(-3):
                phi2 += 2 * pi
            elif phi2 > 2 * pi:
                phi2 -= 2 * pi
        
            C = -(r_min**2 - 1) * (sin(phi2) * (2 * r_min**2 * cos(beta) - 2 * r_min**2 + 1) + 2 * r_min * sin(beta) * cos(phi2))
            D = (r_min**2 - 1) * r_min * (2 * cos(beta) * (r_min**2 * cos(phi2) + r_min**2 - 1) - 2 * r_min**2 * cos(phi2) - 2 * 
                    r_min**2 - 2 * r_min * sin(beta) * sin(phi2) + cos(phi2) + 1)
            # cos(phi1-theta)
            c_phi1_minus_theta = (-2 * r_min**5 + r_min**3 + (-2 * r_min**5 + 3 * r_min**3 - r_min) * cos(phi2) + (2 * r_min**2 - 2 * r_min**4) * 
                                  sin(beta) * sin(phi2) + 4 * (r_min**2 - 1) * r_min**3 * cos(beta) * cos(phi2 / 2)**2 + Alpha[2, 2] * r_min + Alpha[2, 0] * sqrt(1 - r_min**2)) / max(sqrt(C**2 + D**2) , 10**(-8))
            c_phi1_minus_theta = cosine_value_clip(c_phi1_minus_theta)

            E = (r_min**2 - 1) * (cos(beta) * sin(phi2) + r_min * sin(beta) * (cos(phi2) + 1))
            F = (r_min**2 - 1) * (sin(beta) * sin(phi2) - 2 * r_min**3 - 2 * r_min**2 * sin(beta) * sin(phi2) + 2 * (2 * r_min**2 - 1) * r_min * cos(beta) * cos(phi2 / 2)**2 - 
                    2 * (r_min**2 - 1) * r_min * cos(phi2))
            # cos(phi3-gamma)
            c_phi3_minus_sigma = (-2 * r_min**5 + r_min**3 + (-2 * r_min**5 + 3 * r_min**3 - r_min) * cos(phi2) + (2 * r_min**2 - 2 * r_min**4) * sin(beta) * sin(phi2) + 
                                  4 * (r_min**2 - 1) * r_min**3 * cos(beta) * cos(phi2 / 2)**2 + Alpha[2, 2] * r_min + Alpha[0, 2] * sqrt(1 - r_min**2)) / max(sqrt(E**2 + F**2) , 10**(-8))
            c_phi3_minus_sigma = cosine_value_clip(c_phi3_minus_sigma)

            if c_phi1_minus_theta >= -1 and c_phi1_minus_theta <= 1 and c_phi3_minus_sigma >= -1 and c_phi3_minus_sigma <= 1:  # if have solution
                phi1_minus_theta_sol = [acos(c_phi1_minus_theta), 2 * pi - acos(c_phi1_minus_theta)]
                phi3_minus_sigma_sol = [acos(c_phi3_minus_sigma), 2 * pi - acos(c_phi3_minus_sigma)]
                
                for enum1 in phi1_minus_theta_sol:
                    # calculate phi1
                    phi1 = enum1 + atan2(C , D)
                    if abs(phi1-2*pi)<10**(-3):
                        phi1 = 0.0
                    if phi1 < -10**(-3):
                        phi1 += 2 * pi
                    elif phi1 > 2 * pi:
                        phi1 -= 2 * pi
                    
                    for enum2 in phi3_minus_sigma_sol:
                        # calculate phi2
                        phi3 = enum2 + atan2(E , F)
                        if abs(phi3-2*pi)<10**(-3):
                            phi3 = 0.0
                        if phi3 < -10**(-3):
                            phi3 += 2 * pi
                        elif phi3 > 2 * pi:
                            phi3 -= 2 * pi
                        # add candidate paths to list
                        type_list.append({"path": ["L+", "G+", "R+", "R-"], "angles": [phi1, phi2, beta, phi3]})
                        type_list.append({"path": ["R-", "G-", "L-", "L+"], "angles": [phi1, phi2, beta, phi3]})


    # third angle of the R+G+L+|L- and  L-G-R-|R+ paths, which is known
    beta = atan(1/sqrt(U_max**4-1)) + pi/2

    A = -2 * (r_min**2 - 1) * r_min * sin(beta)
    B = -(2 * r_min**4 - 2 * (r_min**2 - 1) * r_min**2 * cos(beta) - 3 * r_min**2 + 1)
    # cos(phi2-gamma)
    c_phi2_minus_gamma = (r_min * (-r_min * (Alpha[2, 2] + 2 * (r_min**2 - 1) * cos(beta) - 2 * r_min**2 + 1) + Alpha[0, 2] * sqrt(1 - r_min**2) + 
                          Alpha[2, 0] * sqrt(1 - r_min**2)) + Alpha[0, 0] * (r_min**2 - 1)) / max(sqrt(A**2 + B**2) , 10**(-8))
    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)
    
    if c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if has solution
        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
        for enum2 in phi2_minus_gamma_sol:
            # calculate phi2
            phi2 = enum2 + atan2(A , B)
            if abs(phi2-2*pi)<10**(-3):
                phi2 = 0.0
            if phi2 < -10**(-3):
                phi2 += 2 * pi
            elif phi2 > 2 * pi:
                phi2 -= 2 * pi
        
            C = -(r_min**2 - 1) * (sin(phi2) * (2 * r_min**2 * cos(beta) - 2 * r_min**2 + 1) + 2 * r_min * sin(beta) * cos(phi2))
            D = (r_min**2 - 1) * r_min * (2 * cos(beta) * (r_min**2 * cos(phi2) + r_min**2 - 1) - 2 * r_min**2 * cos(phi2) - 2 * 
                    r_min**2 - 2 * r_min * sin(beta) * sin(phi2) + cos(phi2) + 1)
            # cos(phi1-theta)
            c_phi1_minus_theta = (-2 * r_min**5 + r_min**3 + (-2 * r_min**5 + 3 * r_min**3 - r_min) * cos(phi2) + (2 * r_min**2 - 2 * r_min**4) * sin(beta) * 
                                  sin(phi2) + 4 * (r_min**2 - 1) * r_min**3 * cos(beta) * cos(phi2 / 2)**2 + Alpha[2, 2] * r_min - Alpha[2, 0] * sqrt(1 - r_min**2)) / max(sqrt(C**2 + D**2) , 10**(-8))
            c_phi1_minus_theta = cosine_value_clip(c_phi1_minus_theta)

            E = (r_min**2 - 1) * (cos(beta) * sin(phi2) + r_min * sin(beta) * (cos(phi2) + 1))
            F = (r_min**2 - 1) * (sin(beta) * sin(phi2) - 2 * r_min**3 - 2 * r_min**2 * sin(beta) * sin(phi2) + 2 * (2 * r_min**2 - 1) * r_min * cos(beta) * cos(phi2 / 2)**2 - 
                    2 * (r_min**2 - 1) * r_min * cos(phi2))
            # cos(phi3-gamma)
            c_phi3_minus_sigma = (-2 * r_min**5 + r_min**3 + (-2 * r_min**5 + 3 * r_min**3 - r_min) * cos(phi2) + (2 * r_min**2 - 2 * r_min**4) * sin(beta) * sin(phi2) + 
                                  4 * (r_min**2 - 1) * r_min**3 * cos(beta) * cos(phi2 / 2)**2 + Alpha[2, 2] * r_min - Alpha[0, 2] * sqrt(1 - r_min**2)) / max(sqrt(E**2 + F**2) , 10**(-8))
            c_phi3_minus_sigma = cosine_value_clip(c_phi3_minus_sigma)

            if c_phi1_minus_theta >= -1 and c_phi1_minus_theta <= 1 and c_phi3_minus_sigma >= -1 and c_phi3_minus_sigma <= 1:  # if have solution
                phi1_minus_theta_sol = [acos(c_phi1_minus_theta), 2 * pi - acos(c_phi1_minus_theta)]
                phi3_minus_sigma_sol = [acos(c_phi3_minus_sigma), 2 * pi - acos(c_phi3_minus_sigma)]
                
                for enum1 in phi1_minus_theta_sol:
                    # calculate phi1
                    phi1 = enum1 + atan2(C , D)
                    if abs(phi1-2*pi)<10**(-3):
                        phi1 = 0.0
                    if phi1 < -10**(-3):
                        phi1 += 2 * pi
                    elif phi1 > 2 * pi:
                        phi1 -= 2 * pi
                    
                    for enum2 in phi3_minus_sigma_sol:
                        # calculate phi2
                        phi3 = enum2 + atan2(E , F)
                        if abs(phi3-2*pi)<10**(-3):
                            phi3 = 0.0
                        if phi3 < -10**(-3):
                            phi3 += 2 * pi
                        elif phi3 > 2 * pi:
                            phi3 -= 2 * pi
                        # add candidate paths to list
                        type_list.append({"path": ["R+", "G+", "L+", "L-"], "angles": [phi1, phi2, beta, phi3]})
                        type_list.append({"path": ["L-", "G-", "R-", "R+"], "angles": [phi1, phi2, beta, phi3]})
    return type_list


def CC_CC_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type CC|CC.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    U_max = sqrt(1/r_min**2 - 1)
    # cosine of second angle of the L+R+|R-L- and R-L-|L+R+ paths
    c_mu_sol1 = (4 * r_min**6 - 6 * r_min**4 + 2 * r_min**2 + sqrt(2) * sqrt(r_min**2 * (r_min**2 - 1)**2 * (Alpha[2, 2] * r_min**2 + Alpha[0, 2] * sqrt(1 - r_min**2) * 
                    r_min - Alpha[2, 0] * sqrt(1 - r_min**2) * r_min + Alpha[0, 0] * (r_min**2 - 1) + 1))) / (4 * r_min**2 * (r_min**2 - 1)**2)
    c_mu_sol2 = (4 * r_min**6 - 6 * r_min**4 + 2 * r_min**2 - sqrt(2) * sqrt(r_min**2 * (r_min**2 - 1)**2 * (Alpha[2, 2] * r_min**2 + Alpha[0, 2] * sqrt(1 - r_min**2) * 
                    r_min - Alpha[2, 0] * sqrt(1 - r_min**2) * r_min + Alpha[0, 0] * (r_min**2 - 1) + 1))) / (4 * r_min**2 * (r_min**2 - 1)**2)
    c_mu_sols=[c_mu_sol1, c_mu_sol2]
    for c_mu in c_mu_sols:
        c_mu = cosine_value_clip(c_mu)
        if c_mu >= -1 and c_mu <= 1:  # if psi has solution
            mu_sol = [acos(c_mu), 2 * pi - acos(c_mu)]
            for mu in mu_sol:
                if mu < atan(1/sqrt(U_max**4-1)) + pi/2+ 0.00001:
                    A = 2 * r_min * (r_min**2 - 1) * sin(mu) * (2 * (r_min**2 - 1) * cos(mu) - 2 * r_min**2 + 1)
                    B = -2 * r_min * (r_min**2 - 1) * ((2 * r_min**2 - 1) * ((r_min**2 - 1) * cos(2 * mu) + 3 * r_min**2 - 1) + (-8 * r_min**4 + 8 * r_min**2 - 1) * cos(mu))
                    # cos(phi1-gamma)
                    c_phi1_minus_gamma = (-12 * r_min**7 + 16 * r_min**5 - 6 * r_min**3 - Alpha[2, 0] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1)**2 * r_min**3 * cos(2 * mu) + 
                                          8 * (2 * r_min**7 - 3 * r_min**5 + r_min**3) * cos(mu) + Alpha[2, 2] * r_min + r_min) / max(sqrt(A**2 + B**2) , 10**(-8))
                    c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                    # cos(phi2-gamma)
                    c_phi2_minus_gamma = (-12 * r_min**7 + 16 * r_min**5 - 6 * r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1)**2 * r_min**3 * cos(2 * mu) + 
                                          8 * (2 * r_min**7 - 3 * r_min**5 + r_min**3) * cos(mu) + Alpha[2, 2] * r_min + r_min) / max(sqrt(A**2 + B**2) , 10**(-8))
                    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)

                    if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if have solution
                        phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
                        
                        for enum1 in phi1_minus_gamma_sol:
                            # calculate phi1
                            phi1 = enum1 + atan2(A , B)
                            if abs(phi1-2*pi)<10**(-3):
                                phi1 = 0.0
                            if phi1 < -10**(-3):
                                phi1 += 2 * pi
                            elif phi1 > 2 * pi:
                                phi1 -= 2 * pi
                            
                            for enum2 in phi2_minus_gamma_sol:
                                # calculate phi2
                                phi2 = enum2 + atan2(A , B)
                                if abs(phi2-2*pi)<10**(-3):
                                    phi2 = 0.0
                                if phi2 < -10**(-3):
                                    phi2 += 2 * pi
                                elif phi2 > 2 * pi:
                                    phi2 -= 2 * pi
                                # add candidate paths to list
                                type_list.append({"path": ["L+", "R+", "R-", "L-"], "angles": [phi1, mu, mu, phi2]})
                                type_list.append({"path": ["R-", "L-", "L+", "R+"], "angles": [phi1, mu, mu, phi2]})

    # cosine of second angle of the R+L+|L-R- and L-R-|R+L+ paths
    c_mu_sol1 = (4 * r_min**6 - 6 * r_min**4 + 2 * r_min**2 + sqrt(2) * sqrt(r_min**2 * (r_min**2 - 1)**2 * (Alpha[2, 2] * r_min**2 - Alpha[0, 2] * sqrt(1 - r_min**2) * 
                    r_min + Alpha[2, 0] * sqrt(1 - r_min**2) * r_min + Alpha[0, 0] * (r_min**2 - 1) + 1))) / (4 * r_min**2 * (r_min**2 - 1)**2)
    c_mu_sol2 = (4 * r_min**6 - 6 * r_min**4 + 2 * r_min**2 - sqrt(2) * sqrt(r_min**2 * (r_min**2 - 1)**2 * (Alpha[2, 2] * r_min**2 - Alpha[0, 2] * sqrt(1 - r_min**2) * 
                    r_min + Alpha[2, 0] * sqrt(1 - r_min**2) * r_min + Alpha[0, 0] * (r_min**2 - 1) + 1))) / (4 * r_min**2 * (r_min**2 - 1)**2)
    c_mu_sols=[c_mu_sol1, c_mu_sol2]
    for c_mu in c_mu_sols:
        c_mu = cosine_value_clip(c_mu)
        if c_mu >= -1 and c_mu <= 1:  # if psi has solution
            mu_sol = [acos(c_mu), 2 * pi - acos(c_mu)]
            for mu in mu_sol:
                if mu < atan(1/sqrt(U_max**4-1)) + pi/2+ 0.00001:
                    A = 2 * r_min * (r_min**2 - 1) * sin(mu) * (2 * (r_min**2 - 1) * cos(mu) - 2 * r_min**2 + 1)
                    B = -2 * r_min * (r_min**2 - 1) * ((2 * r_min**2 - 1) * ((r_min**2 - 1) * cos(2 * mu) + 3 * r_min**2 - 1) + (-8 * r_min**4 + 8 * r_min**2 - 1) * cos(mu))
                    # cos(phi1-gamma)
                    c_phi1_minus_gamma = (-12 * r_min**7 + 16 * r_min**5 - 6 * r_min**3 + Alpha[2, 0] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1)**2 * r_min**3 * cos(2 * mu) + 
                                          8 * (2 * r_min**7 - 3 * r_min**5 + r_min**3) * cos(mu) + Alpha[2, 2] * r_min + r_min) / max(sqrt(A**2 + B**2) , 10**(-8))
                    c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                    # cos(phi2-gamma)
                    c_phi2_minus_gamma = (-12 * r_min**7 + 16 * r_min**5 - 6 * r_min**3 - Alpha[0, 2] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1)**2 * r_min**3 * cos(2 * mu) + 
                                          8 * (2 * r_min**7 - 3 * r_min**5 + r_min**3) * cos(mu) + Alpha[2, 2] * r_min + r_min) / max(sqrt(A**2 + B**2) , 10**(-8))
                    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)

                    if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if have solution
                        phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
                        
                        for enum1 in phi1_minus_gamma_sol:
                            # calculate phi1
                            phi1 = enum1 + atan2(A , B)
                            if abs(phi1-2*pi)<10**(-3):
                                phi1 = 0.0
                            if phi1 < -10**(-3):
                                phi1 += 2 * pi
                            elif phi1 > 2 * pi:
                                phi1 -= 2 * pi
                            
                            for enum2 in phi2_minus_gamma_sol:
                                # calculate phi2
                                phi2 = enum2 + atan2(A , B)
                                if abs(phi2-2*pi)<10**(-3):
                                    phi2 = 0.0
                                if phi2 < -10**(-3):
                                    phi2 += 2 * pi
                                elif phi2 > 2 * pi:
                                    phi2 -= 2 * pi
                                # add candidate paths to list
                                type_list.append({"path": ["R+", "L+", "L-", "R-"], "angles": [phi1, mu, mu, phi2]})
                                type_list.append({"path": ["L-", "R-", "R+", "L+"], "angles": [phi1, mu, mu, phi2]})
    return type_list

    
def C_CGC_C_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type C|CGC|C.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    U_max = sqrt(1/r_min**2 - 1)
    # second and fourth angle of the L+|L-G-L-|L+ and R-|R+G+R+|R- paths, which is known
    beta = atan(1/sqrt(U_max**4-1)) + pi/2

    A = 4 * (r_min**2 - 1) * r_min * sin(beta) * (2 * r_min**2 * cos(beta) - 2 * r_min**2 + 1)
    B = -(r_min**2 - 1) * (6 * r_min**4 - 6 * r_min**2 + (4 * r_min**2 - 8 * r_min**4) * cos(beta) + 2 * (r_min**4 + r_min**2) * cos(2 * beta) + 1)
    
    # cos(phi2-gamma)
    c_phi2_minus_gamma = (r_min * (r_min * (Alpha[2, 2] - (2 * (r_min**2 - 1) * cos(beta) - 2 * r_min**2 + 1)**2) + Alpha[0, 2] * sqrt(1 - r_min**2) + 
                            Alpha[2, 0] * sqrt(1 - r_min**2)) - Alpha[0, 0] * (r_min**2 - 1)) / max(sqrt(A**2 + B**2) , 10**(-8))
    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)
    if c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if has solution
        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
        for enum2 in phi2_minus_gamma_sol:
            # calculate phi2
            phi2 = enum2 + atan2(A , B)
            if abs(phi2-2*pi)<10**(-3):
                phi2 = 0.0
            if phi2 < -10**(-3):
                phi2 += 2 * pi
            elif phi2 > 2 * pi:
                phi2 -= 2 * pi
            C = (r_min**5 + r_min) * sin(2 * beta) - (r_min**2 - 1) * sin(phi2) * (cos(beta) - 2 * r_min**2 * cos(beta) + 2 * r_min**2 * cos(2 * beta)
                    ) + r_min * sin(beta) * ((2 * r_min**4 - 3 * r_min**2 + 1) * (cos(phi2) - 1) - 2 * cos(beta) * ((r_min**4 - 1) * cos(phi2) + 2 * r_min**2))

            D = (4 * r_min**7 - 6 * r_min**5 - 6 * r_min**4 * sin(2  * beta) * sin(phi2) + 2 * r_min**3 + 2 * (r_min**2 - 1)**2 * (2 * r_min**2 - 1) * r_min * cos(beta)**2 
                    + (r_min**2 - 1) * r_min * cos(phi2) * ((8 * r_min**4 - 8 * r_min**2 + 1) * cos(beta) - (2 * r_min**2 - 1) * ((r_min**2 + 1) * cos(2 * beta) 
                    + 3 * (r_min**2 - 1))) + (-8 * r_min**7 + 16 * r_min**5 - 9 * r_min**3 + r_min) * cos(beta) + sin(beta) * sin(phi2) * 
                    (-8 * r_min**6 + 16 * r_min**4 - 9 * r_min**2 + 4 * (2 * r_min**6 + r_min**2) * cos(beta) + 1))
            # cos(phi1-theta)
            c_phi1_minus_theta = (4 * r_min**7 - 4 * r_min**5 + r_min**3 - Alpha[2, 0] * sqrt(1 - r_min**2) + 4 * (r_min**2 - 1) * r_min**2 * 
                                  sin(beta) * sin(phi2) * (2 * r_min**2 * cos(beta) - 2 * r_min**2 + 1) - (r_min**2 - 1) * r_min * cos(phi2) * 
                                  (6 * r_min**4 - 6 * r_min**2 + (4 * r_min**2 - 8 * r_min**4) * cos(beta) + 2 * (r_min**4 + r_min**2) * cos(2 * beta) + 1) 
                                  + 4 * (r_min**2 - 1)**2 * r_min**3 * cos(beta)**2 + 4 * (1 - 2 * r_min**2) * (r_min**2 - 1) * r_min**3 * cos(beta) 
                                  - Alpha[2, 2] * r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
            c_phi1_minus_theta = cosine_value_clip(c_phi1_minus_theta)
            # cos(phi3-theta)
            c_phi3_minus_theta = (4 * r_min**7 - 4 * r_min**5 + r_min**3 - Alpha[0, 2] * sqrt(1 - r_min**2) + 4 * (r_min**2 - 1) * r_min**2 * 
                                  sin(beta) * sin(phi2) * (2 * r_min**2 * cos(beta) - 2 * r_min**2 + 1) - (r_min**2 - 1) * r_min * cos(phi2) * 
                                  (6 * r_min**4 - 6 * r_min**2 + (4 * r_min**2 - 8 * r_min**4) * cos(beta) + 2 * (r_min**4 + r_min**2) * cos(2 * beta) + 1) 
                                  + 4 * (r_min**2 - 1)**2 * r_min**3 * cos(beta)**2 + 4 * (1 - 2 * r_min**2) * (r_min**2 - 1) * r_min**3 * cos(beta) 
                                  - Alpha[2, 2] * r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
            c_phi3_minus_theta = cosine_value_clip(c_phi3_minus_theta)

            if c_phi1_minus_theta >= -1 and c_phi1_minus_theta <= 1 and c_phi3_minus_theta >= -1 and c_phi3_minus_theta <= 1:  # if have solution
                phi1_minus_theta_sol = [acos(c_phi1_minus_theta), 2 * pi - acos(c_phi1_minus_theta)]
                phi3_minus_theta_sol = [acos(c_phi3_minus_theta), 2 * pi - acos(c_phi3_minus_theta)]
                
                for enum1 in phi1_minus_theta_sol:
                    # calculate phi1
                    phi1 = enum1 + atan2(C , D)
                    if abs(phi1-2*pi)<10**(-3):
                        phi1 = 0.0
                    if phi1 < -10**(-3):
                        phi1 += 2 * pi
                    elif phi1 > 2 * pi:
                        phi1 -= 2 * pi
                    for enum2 in phi3_minus_theta_sol:
                        # calculate phi2
                        phi3 = enum2 + atan2(C , D)
                        if abs(phi3-2*pi)<10**(-3):
                            phi3 = 0.0
                        if phi3 < -10**(-3):
                            phi3 += 2 * pi
                        elif phi3 > 2 * pi:
                            phi3 -= 2 * pi
                        # add candidate paths to list
                        type_list.append({"path": ["L+", "L-", "G-", "L-", "L+"], "angles": [phi1, beta, phi2, beta, phi3]})
                        type_list.append({"path": ["R-", "R+", "G+", "R+", "R-"], "angles": [phi1, beta, phi2, beta, phi3]})

    # second and fourth angle of the R+|R-G-R-|R+ and L-L+G+L+L- paths, which is known
    beta = atan(1/sqrt(U_max**4-1)) + pi/2

    A = 4 * (r_min**2 - 1) * r_min * sin(beta) * (2 * r_min**2 * cos(beta) - 2 * r_min**2 + 1)
    B = -(r_min**2 - 1) * (6 * r_min**4 - 6 * r_min**2 + (4 * r_min**2 - 8 * r_min**4) * cos(beta) + 2 * (r_min**4 + r_min**2) * cos(2 * beta) + 1)
    # cos(phi2-gamma)
    c_phi2_minus_gamma = -(r_min * (r_min * ((2 * (r_min**2 - 1) * cos(beta) - 2 * r_min**2 + 1)**2 - Alpha[2, 2]) + Alpha[0, 2] * 
                            sqrt(1 - r_min**2) + Alpha[2, 0] * sqrt(1 - r_min**2)) + Alpha[0, 0] * (r_min**2 - 1)) / max(sqrt(A**2 + B**2) , 10**(-8))
    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)
    
    if c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if has solution
        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
        for enum2 in phi2_minus_gamma_sol:
            # calculate phi2
            phi2 = enum2 + atan2(A , B)
            if abs(phi2-2*pi)<10**(-3):
                phi2 = 0.0
            if phi2 < -10**(-3):
                phi2 += 2 * pi
            elif phi2 > 2 * pi:
                phi2 -= 2 * pi
        
            C = (r_min**5 + r_min) * sin(2 * beta) - (r_min**2 - 1) * sin(phi2) * (cos(beta) - 2 * r_min**2 * cos(beta) + 2 * r_min**2 * cos(2 * beta)
                    ) + r_min * sin(beta) * ((2 * r_min**4 - 3 * r_min**2 + 1) * (cos(phi2) - 1) - 2 * cos(beta) * ((r_min**4 - 1) * cos(phi2) + 2 * r_min**2))

            D = (4 * r_min**7 - 6 * r_min**5 - 6 * r_min**4 * sin(2 * beta) * sin(phi2) + 2 * r_min**3 + 2 * (r_min**2 - 1)**2 * (2 * r_min**2 - 1) * r_min * cos(beta)**2 
                    + (r_min**2 - 1) * r_min * cos(phi2) * ((8 * r_min**4 - 8 * r_min**2 + 1) * cos(beta) - (2 * r_min**2 - 1) * ((r_min**2 + 1) * cos(2 * beta) 
                    + 3 * (r_min**2 - 1))) + (-8 * r_min**7 + 16 * r_min**5 - 9 * r_min**3 + r_min) * cos(beta) + sin(beta) * sin(phi2) * 
                    (-8 * r_min**6 + 16 * r_min**4 - 9 * r_min**2 + 4 * (2 * r_min**6 + r_min**2) * cos(beta) + 1))
            # cos(phi1-theta)
            c_phi1_minus_theta = (4 * r_min**7 - 4 * r_min**5 + r_min**3 + Alpha[2, 0] * sqrt(1 - r_min**2) + 4 * (r_min**2 - 1) * r_min**2 * sin(beta) * 
                                  sin(phi2) * (2 * r_min**2 * cos(beta) - 2 * r_min**2 + 1) - (r_min**2 - 1) * r_min * cos(phi2) * (6 * r_min**4 - 6 * r_min**2 + 
                                  (4 * r_min**2 - 8 * r_min**4) * cos(beta) + 2 * (r_min**4 + r_min**2) * cos(2 * beta) + 1) + 4 * (r_min**2 - 1)**2 * 
                                  r_min**3 * cos(beta)**2 + 4 * (1 - 2 * r_min**2) * (r_min**2 - 1) * r_min**3 * cos(beta) - Alpha[2, 2] * r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
            c_phi1_minus_theta = cosine_value_clip(c_phi1_minus_theta)
            # cos(phi3-theta)
            c_phi3_minus_theta = (4 * r_min**7 - 4 * r_min**5 + r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) + 4 * (r_min**2 - 1) * r_min**2 * 
                                  sin(beta) * sin(phi2) * (2 * r_min**2 * cos(beta) - 2 * r_min**2 + 1) - (r_min**2 - 1) * r_min * cos(phi2) * 
                                  (6 * r_min**4 - 6 * r_min**2 + (4 * r_min**2 - 8 * r_min**4) * cos(beta) + 2 * (r_min**4 + r_min**2) * cos(2 * beta) + 1) 
                                  + 4 * (r_min**2 - 1)**2 * r_min**3 * cos(beta)**2 + 4 * (1 - 2 * r_min**2) * (r_min**2 - 1) * r_min**3 * cos(beta) 
                                  - Alpha[2, 2] * r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
            c_phi3_minus_theta = cosine_value_clip(c_phi3_minus_theta)

            if c_phi1_minus_theta >= -1 and c_phi1_minus_theta <= 1 and c_phi3_minus_theta >= -1 and c_phi3_minus_theta <= 1:  # if have solution
                phi1_minus_theta_sol = [acos(c_phi1_minus_theta), 2 * pi - acos(c_phi1_minus_theta)]
                phi3_minus_theta_sol = [acos(c_phi3_minus_theta), 2 * pi - acos(c_phi3_minus_theta)]
                
                for enum1 in phi1_minus_theta_sol:
                    # calculate phi1
                    phi1 = enum1 + atan2(C , D)
                    if abs(phi1-2*pi)<10**(-3):
                        phi1 = 0.0
                    if phi1 < -10**(-3):
                        phi1 += 2 * pi
                    elif phi1 > 2 * pi:
                        phi1 -= 2 * pi
                    
                    for enum2 in phi3_minus_theta_sol:
                        # calculate phi2
                        phi3 = enum2 + atan2(C , D)
                        if abs(phi3-2*pi)<10**(-3):
                            phi3 = 0.0
                        if phi3 < -10**(-3):
                            phi3 += 2 * pi
                        elif phi3 > 2 * pi:
                            phi3 -= 2 * pi
                        # add candidate paths to list
                        type_list.append({"path": ["R+", "R-", "G-", "R-", "R+"], "angles": [phi1, beta, phi2, beta, phi3]})
                        type_list.append({"path": ["L-", "L+", "G+", "L+", "L-"], "angles": [phi1, beta, phi2, beta, phi3]})
    
    # second and fourth angle of the L+|L-G-R-|R+ and "R-|R+G+L+|L-" paths, which is known
    beta = atan(1/sqrt(U_max**4-1)) + pi/2

    A = 4 * r_min * (r_min**2 - 1) * (2 * r_min**2 - 1) * sin(beta) - 8 * r_min**3 * (r_min**2 - 1) * sin(beta) * cos(beta)
    B = -4 * (r_min**2 - 1) * (2 * r_min**2 - 1) * r_min**2 * cos(beta) + 4 * (r_min**2 - 1) * r_min**4 * cos(beta)**2 + (r_min**2 - 
        1) * (4 * r_min**4 + 2 * r_min**2 * cos(2 * beta) - 6 * r_min**2 + 1)
    # cos(phi2-gamma)
    c_phi2_minus_gamma = (r_min * (r_min * (Alpha[2, 2] - (2 * (r_min**2 - 1) * cos(beta) - 2 * r_min**2 + 1)**2) + Alpha[0, 2] * 
                            sqrt(1 - r_min**2) - Alpha[2, 0] * sqrt(1 - r_min**2)) + Alpha[0, 0] * (r_min**2 - 1)) / max(sqrt(A**2 + B**2) , 10**(-8))
    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)
    
    if c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if has solution
        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
        for enum2 in phi2_minus_gamma_sol:
            # calculate phi2
            phi2 = enum2 + atan2(A , B)
            if abs(phi2-2*pi)<10**(-3):
                phi2 = 0.0
            if phi2 < -10**(-3):
                phi2 += 2 * pi
            elif phi2 > 2 * pi:
                phi2 -= 2 * pi
        
            C = (r_min**5 + r_min) * sin(2 * beta) + (r_min**2 - 1) * sin(phi2) * (cos(beta) - 2 * r_min**2 * cos(beta) + 2 * r_min**2 * cos(2 * beta)) + r_min * sin(
                 beta) * (2 * cos(beta) * ((r_min**4 - 1) * cos(phi2) - 2 * r_min**2) - (2 * r_min**4 - 3 * r_min**2 + 1) * (cos(phi2) + 1))

            D = 4 * r_min**7 - 6 * r_min**5 + 6 * r_min**4 * sin(2 * beta) * sin(phi2) + 2 * r_min**3 + 2 * (r_min**2 - 1)**2 * (2 * r_min**2 - 1) * r_min * cos(
                 beta)**2 + (r_min**2 - 1) * r_min * cos(phi2) * ((2 * r_min**2 - 1) * ((r_min**2 + 1) * cos(2 * beta) + 3 * (r_min**2 - 1)) + (-8 * r_min**4 + 8 * r_min**2 
                 - 1) * cos(beta)) + (-8 * r_min**7 + 16 * r_min**5 - 9 * r_min**3 + r_min) * cos(beta) + sin(beta) * sin(phi2) * (8 * r_min**6 - 16 * r_min**4 + 
                 9 * r_min**2 - 4 * (2 * r_min**6 + r_min**2) * cos(beta) - 1)
            # cos(phi1-theta)
            c_phi1_minus_theta = (4 * r_min**7 - 4 * r_min**5 + r_min**3 + Alpha[2,0] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1) * r_min**4 * sin(2 * beta) * sin(phi2)
                                  + (r_min**2 - 1) * r_min * cos(phi2) * (6 * r_min**4 - 6 * r_min**2 + 2 * (r_min**4 + r_min**2) * cos(2 * beta) + 1) + 
                                  4 * (r_min**2 - 1)**2 * r_min**3 * cos(beta)**2 + 4 * (2 * r_min**6 - 3 * r_min**4 + r_min**2) * sin(beta) * sin(phi2) - 
                                  8 * (2 * r_min**4 - 3 * r_min**2 + 1) * r_min**3 * cos(beta) * cos(phi2 / 2)**2 - Alpha[2, 2] * r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
            c_phi1_minus_theta = cosine_value_clip(c_phi1_minus_theta)
            # cos(phi3-theta)
            c_phi3_minus_theta = (4 * r_min**7 - 4 * r_min**5 + r_min**3 - Alpha[0, 2] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1) * r_min**4 * sin(2 * beta) * 
                                  sin(phi2) + (r_min**2 - 1) * r_min * cos(phi2) * (6 * r_min**4 - 6 * r_min**2 + 2 * (r_min**4 + r_min**2) * cos(2 * beta) + 
                                  1) + 4 * (r_min**2 - 1)**2 * r_min**3 * cos(beta)**2 + 4 * (2 * r_min**6 - 3 * r_min**4 + r_min**2) * sin(beta) * sin(phi2) 
                                  - 8 * (2 * r_min**4 - 3 * r_min**2 + 1) * r_min**3 * cos(beta) * cos(phi2 / 2)**2 - Alpha[2, 2] * r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
            c_phi3_minus_theta = cosine_value_clip(c_phi3_minus_theta)
            if c_phi1_minus_theta >= -1 and c_phi1_minus_theta <= 1 and c_phi3_minus_theta >= -1 and c_phi3_minus_theta <= 1:  # if have solution
                phi1_minus_theta_sol = [acos(c_phi1_minus_theta), 2 * pi - acos(c_phi1_minus_theta)]
                phi3_minus_theta_sol = [acos(c_phi3_minus_theta), 2 * pi - acos(c_phi3_minus_theta)]
                
                for enum1 in phi1_minus_theta_sol:
                    # calculate phi1
                    phi1 = enum1 + atan2(C , D)
                    if abs(phi1-2*pi)<10**(-3):
                        phi1 = 0.0
                    if phi1 < -10**(-3):
                        phi1 += 2 * pi
                    elif phi1 > 2 * pi:
                        phi1 -= 2 * pi
                    
                    for enum2 in phi3_minus_theta_sol:
                        # calculate phi2
                        phi3 = enum2 + atan2(C , D)
                        if abs(phi3-2*pi)<10**(-3):
                            phi3 = 0.0
                        if phi3 < -10**(-3):
                            phi3 += 2 * pi
                        elif phi3 > 2 * pi:
                            phi3 -= 2 * pi
                        # add candidate paths to list
                        type_list.append({"path": ["L+", "L-", "G-", "R-", "R+"], "angles": [phi1, beta, phi2, beta, phi3]})
                        type_list.append({"path": ["R-", "R+", "G+", "L+", "L-"], "angles": [phi1, beta, phi2, beta, phi3]})


    # second and fourth angle of the R+|R-G-L-|L+ and L-|L+G+R+|R- paths, which is known
    beta = atan(1/sqrt(U_max**4-1)) + pi/2

    A = 4 * r_min * (r_min**2 - 1) * (2 * r_min**2 - 1) * sin(beta) - 8 * r_min**3 * (r_min**2 - 1) * sin(beta) * cos(beta)
    B = -4 * (r_min**2 - 1) * (2 * r_min**2 - 1) * r_min**2 * cos(beta) + 4 * (r_min**2 - 1) * r_min**4 * cos(beta)**2 + (r_min**2 - 
        1) * (4 * r_min**4 + 2 * r_min**2 * cos(2 * beta) - 6 * r_min**2 + 1)
    # cos(phi2-gamma)
    c_phi2_minus_gamma = (r_min * (r_min * (Alpha[2, 2] - (2 * (r_min**2 - 1) * cos(beta) - 2 * r_min**2 + 1)**2) - Alpha[0, 2] * 
                            sqrt(1 - r_min**2) + Alpha[2, 0] * sqrt(1 - r_min**2)) + Alpha[0, 0] * (r_min**2 - 1)) / max(sqrt(A**2 + B**2) , 10**(-8))
    c_phi2_minus_gamma = cosine_value_clip(c_phi2_minus_gamma)
    
    if c_phi2_minus_gamma >= -1 and c_phi2_minus_gamma <= 1:  # if has solution
        phi2_minus_gamma_sol = [acos(c_phi2_minus_gamma), 2 * pi - acos(c_phi2_minus_gamma)]
        for enum2 in phi2_minus_gamma_sol:
            # calculate phi2
            phi2 = enum2 + atan2(A , B)
            if abs(phi2-2*pi)<10**(-3):
                phi2 = 0.0
            if phi2 < -10**(-3):
                phi2 += 2 * pi
            elif phi2 > 2 * pi:
                phi2 -= 2 * pi
        
            C = (r_min**5 + r_min) * sin(2 * beta) + (r_min**2 - 1) * sin(phi2) * (cos(beta) - 2 * r_min**2 * cos(beta) + 2 * r_min**2 * cos(2 * beta)) + r_min * sin(
                 beta) * (2 * cos(beta) * ((r_min**4 - 1) * cos(phi2) - 2 * r_min**2) - (2 * r_min**4 - 3 * r_min**2 + 1) * (cos(phi2) + 1))

            D = 4 * r_min**7 - 6 * r_min**5 + 6 * r_min**4 * sin(2 * beta) * sin(phi2) + 2 * r_min**3 + 2 * (r_min**2 - 1)**2 * (2 * r_min**2 - 1) * r_min * cos(
                 beta)**2 + (r_min**2 - 1) * r_min * cos(phi2) * ((2 * r_min**2 - 1) * ((r_min**2 + 1) * cos(2 * beta) + 3 * (r_min**2 - 1)) + (-8 * r_min**4 + 8 * r_min**2 
                 - 1) * cos(beta)) + (-8 * r_min**7 + 16 * r_min**5 - 9 * r_min**3 + r_min) * cos(beta) + sin(beta) * sin(phi2) * (8 * r_min**6 - 16 * r_min**4 + 
                 9 * r_min**2 - 4 * (2 * r_min**6 + r_min**2) * cos(beta) - 1)
            # cos(phi1-theta)
            c_phi1_minus_theta = (4 * r_min**7 - 4 * r_min**5 + r_min**3 - Alpha[2,0] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1) * r_min**4 * sin(2 * beta) * sin(phi2)
                                  + (r_min**2 - 1) * r_min * cos(phi2) * (6 * r_min**4 - 6 * r_min**2 + 2 * (r_min**4 + r_min**2) * cos(2 * beta) + 1) + 
                                  4 * (r_min**2 - 1)**2 * r_min**3 * cos(beta)**2 + 4 * (2 * r_min**6 - 3 * r_min**4 + r_min**2) * sin(beta) * sin(phi2) - 
                                  8 * (2 * r_min**4 - 3 * r_min**2 + 1) * r_min**3 * cos(beta) * cos(phi2 / 2)**2 - Alpha[2, 2] * r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
            c_phi1_minus_theta = cosine_value_clip(c_phi1_minus_theta)
            # cos(phi3-theta)
            c_phi3_minus_theta = (4 * r_min**7 - 4 * r_min**5 + r_min**3 + Alpha[0, 2] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1) * r_min**4 * sin(2 * beta) * 
                                  sin(phi2) + (r_min**2 - 1) * r_min * cos(phi2) * (6 * r_min**4 - 6 * r_min**2 + 2 * (r_min**4 + r_min**2) * cos(2 * beta) + 
                                  1) + 4 * (r_min**2 - 1)**2 * r_min**3 * cos(beta)**2 + 4 * (2 * r_min**6 - 3 * r_min**4 + r_min**2) * sin(beta) * sin(phi2) 
                                  - 8 * (2 * r_min**4 - 3 * r_min**2 + 1) * r_min**3 * cos(beta) * cos(phi2 / 2)**2 - Alpha[2, 2] * r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
            c_phi3_minus_theta = cosine_value_clip(c_phi3_minus_theta)
            if c_phi1_minus_theta >= -1 and c_phi1_minus_theta <= 1 and c_phi3_minus_theta >= -1 and c_phi3_minus_theta <= 1:  # if have solution
                phi1_minus_theta_sol = [acos(c_phi1_minus_theta), 2 * pi - acos(c_phi1_minus_theta)]
                phi3_minus_theta_sol = [acos(c_phi3_minus_theta), 2 * pi - acos(c_phi3_minus_theta)]
                
                for enum1 in phi1_minus_theta_sol:
                    # calculate phi1
                    phi1 = enum1 + atan2(C , D)
                    if abs(phi1-2*pi)<10**(-3):
                        phi1 = 0.0
                    if phi1 < -10**(-3):
                        phi1 += 2 * pi
                    elif phi1 > 2 * pi:
                        phi1 -= 2 * pi
                    
                    for enum2 in phi3_minus_theta_sol:
                        # calculate phi2
                        phi3 = enum2 + atan2(C , D)
                        if abs(phi3-2*pi)<10**(-3):
                            phi3 = 0.0
                        if phi3 < -10**(-3):
                            phi3 += 2 * pi
                        elif phi3 > 2 * pi:
                            phi3 -= 2 * pi
                        # add candidate paths to list
                        type_list.append({"path": ["R+", "R-", "G-", "L-", "L+"], "angles": [phi1, beta, phi2, beta, phi3]})
                        type_list.append({"path": ["L-", "L+", "G+", "R+", "R-"], "angles": [phi1, beta, phi2, beta, phi3]})
    return type_list


def C_CC_CC_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type C|CC|CC.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    U_max = sqrt(1/r_min**2 - 1)
    # cosine of second, third, and fourth angle of the L+|L-R-|R+L+ and R-|R+L+|L-R- paths
    # Define the coefficients for the cubic equation in cos(mu)
    a3 = -16 * r_min**8 + 32 * r_min**6 - 16 * r_min**4
    a2 = 48 * r_min**8 - 96 * r_min**6 + 56 * r_min**4 - 8 * r_min**2
    a1 = -48 * r_min**8 + 96 * r_min**6 - 64 * r_min**4 + 16 * r_min**2
    a0 = 16 * r_min**8 - 32 * r_min**6 + 24 * r_min**4 - 8 * r_min**2 + 1 - r_min * (Alpha[0, 2] * sqrt(1 - r_min**2) + Alpha[2, 0] * 
         sqrt(1 - r_min**2) + Alpha[2, 2] * r_min) + Alpha[0, 0] * (r_min**2 - 1)

    # Form the polynomial coefficients
    coefficients = [a3, a2, a1, a0]
    # Solve the cubic equation
    roots = np.roots(coefficients)
    
    # Filter for real roots
    c_mu_sols = roots[np.isreal(roots)].real
    #c_mu_sols = [z.real for z in roots]
    #print(c_mu_sols)
    for c_mu in c_mu_sols:
        c_mu = cosine_value_clip(c_mu)
        if c_mu >= -1 and c_mu <= 1:  # if has solution
            mu_sol = [acos(c_mu), 2 * pi - acos(c_mu)]
            for mu in mu_sol:
                if mu < atan(1/sqrt(U_max**4-1)) + pi/2+ 0.00001:
                    A = 8 * (r_min**2 - 1) * r_min**3 * sin(mu / 2)**2 * sin(mu) * (2 * (r_min**2 - 1) * cos(mu) - 2 * r_min**2 + 1)
                    B = 8 * (r_min**2 - 1) * r_min * sin(mu / 2)**2 * ((r_min**2 - 1) * (6 * r_min**4 + (2 * r_min**2 - 1) * r_min**2 * 
                         cos(2 * mu) - 3 * r_min**2 + 1) - r_min**2 * (8 * r_min**4 - 12 * r_min**2 + 5) * cos(mu))
                    # cos(phi1-gamma)
                    c_phi1_minus_gamma = (40 * r_min**9 - 80 * r_min**7 + 52 * r_min**5 - 12 * r_min**3 - Alpha[2,0] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1)**2 * 
                                          r_min**5 * cos(3 * mu) - 4 * (15 * r_min**6 - 30 * r_min**4 + 19 * r_min**2 - 4) * r_min**3 * cos(mu) + 
                                          4 * (6 * r_min**6 - 12 * r_min**4 + 7 * r_min**2 - 1) * r_min**3 * cos(2 * mu) - Alpha[2,2] * r_min + r_min) / max(sqrt(A**2 + B**2) , 10**(-15))
                    #print(c_phi1_minus_gamma)
                    c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                    #print(c_phi1_minus_gamma)

                    C = -8 * (r_min**2 - 1)**2 * r_min * sin(mu / 2)**2 * sin(mu) * (2 * r_min**2 * cos(mu) - 2 * r_min**2 + 1)
                    D = 8 * (r_min**2 - 1) * r_min * sin(mu / 2)**2 * (r_min**2 * (6 * r_min**4 - 9 * r_min**2 + (2 * r_min**4 - 3 * r_min**2 + 1) * cos(2 * mu) + 4) + 
                        (-8 * r_min**6 + 12 * r_min**4 - 5 * r_min**2 + 1) * cos(mu))
                    # cos(phi2-sigma)
                    c_phi2_minus_sigma = (40 * r_min**9 - 80 * r_min**7 + 52 * r_min**5 - 12 * r_min**3 - Alpha[0,2] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1)**2 * 
                                          r_min**5 * cos(3 * mu) - 4 * (15 * r_min**6 - 30 * r_min**4 + 19 * r_min**2 - 4) * r_min**3 * cos(mu) + 4 * (6 * r_min**6 - 
                                          12 * r_min**4 + 7 * r_min**2 - 1) * r_min**3 * cos(2 * mu) - Alpha[2,2] * r_min + r_min) / max(sqrt(C**2 + D**2) , 10**(-15))
                    c_phi2_minus_sigma = cosine_value_clip(c_phi2_minus_sigma)
                    if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi2_minus_sigma >= -1 and c_phi2_minus_sigma <= 1:  # if have solution
                        phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                        phi2_minus_sigma_sol = [acos(c_phi2_minus_sigma), 2 * pi - acos(c_phi2_minus_sigma)]
                        for enum1 in phi1_minus_gamma_sol:
                            
                            # calculate phi1
                            phi1 = enum1 + atan2(A , B)
                            if abs(phi1-2*pi)<10**(-3):
                                phi1 = 0.0
                            if phi1 < -10**(-3):
                                phi1 += 2 * pi
                            elif phi1 > 2 * pi:
                                phi1 -= 2 * pi
                            
                            for enum2 in phi2_minus_sigma_sol:
                                
                                # calculate phi2
                                phi2 = enum2 + atan2(C , D)
                                if abs(phi2-2*pi)<10**(-3):
                                    phi2 = 0.0
                                if phi2 < -10**(-3):
                                    phi2 += 2 * pi
                                elif phi2 > 2 * pi:
                                    phi2 -= 2 * pi
                                # add candidate paths to list
                                type_list.append({"path": ["L+", "L-", "R-", "R+", "L+"], "angles": [phi1, mu, mu, mu, phi2]})
                                type_list.append({"path": ["R-", "R+", "L+", "L-", "R-"], "angles": [phi1, mu, mu, mu, phi2]})
                                

    # cosine of second, third, and fourth angle of the R+|R-L-|L+R+ and L-|L+R+|R-L- paths
    # Define the coefficients for the cubic equation in cos(mu)
    a3 = -16 * r_min**8 + 32 * r_min**6 - 16 * r_min**4
    a2 = 48 * r_min**8 - 96 * r_min**6 + 56 * r_min**4 - 8 * r_min**2
    a1 = -48 * r_min**8 + 96 * r_min**6 - 64 * r_min**4 + 16 * r_min**2
    a0 = 16 * r_min**8 - 32 * r_min**6 + 24 * r_min**4 - 8 * r_min**2 + 1 - r_min * (-Alpha[0, 2] * sqrt(1 - r_min**2) - Alpha[2, 0] * 
         sqrt(1 - r_min**2) + Alpha[2, 2] * r_min) + Alpha[0, 0] * (r_min**2 - 1)

    # Form the polynomial coefficients
    coefficients = [a3, a2, a1, a0]
    # Solve the cubic equation
    roots = np.roots(coefficients)
    
    # Filter for real roots
    c_mu_sols = roots[np.isreal(roots)].real
    #c_mu_sols = [z.real for z in roots]
    #print(c_mu_sols)
    for c_mu in c_mu_sols:
        #print(c_mu)
        c_mu = cosine_value_clip(c_mu)
        #print(c_mu)
        if c_mu >= -1 and c_mu <= 1:  # if has solution
            mu_sol = [acos(c_mu), 2 * pi - acos(c_mu)]
            #print(mu_sol)
            for mu in mu_sol:
                if mu < atan(1/sqrt(U_max**4-1)) + pi/2+ 0.00001:
                    A = 8 * (r_min**2 - 1) * r_min**3 * sin(mu / 2)**2 * sin(mu) * (2 * (r_min**2 - 1) * cos(mu) - 2 * r_min**2 + 1)
                    B = 8 * (r_min**2 - 1) * r_min * sin(mu / 2)**2 * ((r_min**2 - 1) * (6 * r_min**4 + (2 * r_min**2 - 1) * r_min**2 * 
                         cos(2 * mu) - 3 * r_min**2 + 1) - r_min**2 * (8 * r_min**4 - 12 * r_min**2 + 5) * cos(mu))
                    # cos(phi1-gamma)
                    c_phi1_minus_gamma = (40 * r_min**9 - 80 * r_min**7 + 52 * r_min**5 - 12 * r_min**3 + Alpha[2,0] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1)**2 * 
                                          r_min**5 * cos(3 * mu) - 4 * (15 * r_min**6 - 30 * r_min**4 + 19 * r_min**2 - 4) * r_min**3 * cos(mu) + 
                                          4 * (6 * r_min**6 - 12 * r_min**4 + 7 * r_min**2 - 1) * r_min**3 * cos(2 * mu) - Alpha[2,2] * r_min + r_min) / max(sqrt(A**2 + B**2) , 10**(-8))
                    c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                    C = -8 * (r_min**2 - 1)**2 * r_min * sin(mu / 2)**2 * sin(mu) * (2 * r_min**2 * cos(mu) - 2 * r_min**2 + 1)
                    D = 8 * (r_min**2 - 1) * r_min * sin(mu / 2)**2 * (r_min**2 * (6 * r_min**4 - 9 * r_min**2 + (2 * r_min**4 - 3 * r_min**2 + 1) * cos(2 * mu) + 4) + 
                        (-8 * r_min**6 + 12 * r_min**4 - 5 * r_min**2 + 1) * cos(mu))
                    # cos(phi2-sigma)
                    c_phi2_minus_sigma = (40 * r_min**9 - 80 * r_min**7 + 52 * r_min**5 - 12 * r_min**3 + Alpha[0,2] * sqrt(1 - r_min**2) - 4 * (r_min**2 - 1)**2 * 
                                          r_min**5 * cos(3 * mu) - 4 * (15 * r_min**6 - 30 * r_min**4 + 19 * r_min**2 - 4) * r_min**3 * cos(mu) + 4 * (6 * r_min**6 - 
                                          12 * r_min**4 + 7 * r_min**2 - 1) * r_min**3 * cos(2 * mu) - Alpha[2,2] * r_min + r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
                    c_phi2_minus_sigma = cosine_value_clip(c_phi2_minus_sigma)

                    if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi2_minus_sigma >= -1 and c_phi2_minus_sigma <= 1:  # if have solution
                        phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                        phi2_minus_sigma_sol = [acos(c_phi2_minus_sigma), 2 * pi - acos(c_phi2_minus_sigma)]
                        for enum1 in phi1_minus_gamma_sol:
                            # calculate phi1
                            phi1 = enum1 + atan2(A , B)
                            if abs(phi1-2*pi)<10**(-3):
                                phi1 = 0.0
                            if phi1 < -10**(-3):
                                phi1 += 2 * pi
                            elif phi1 > 2 * pi:
                                phi1 -= 2 * pi
                            for enum2 in phi2_minus_sigma_sol:
                                # calculate phi2
                                phi2 = enum2 + atan2(C , D)
                                if abs(phi2-2*pi)<10**(-3):
                                    phi2 = 0.0
                                if phi2 < -10**(-3):
                                    phi2 += 2 * pi
                                elif phi2 > 2 * pi:
                                    phi2 -= 2 * pi
                                # add candidate paths to list
                                type_list.append({"path": ["R+", "R-", "L-", "L+", "R+"], "angles": [phi1, mu, mu, mu, phi2]})
                                type_list.append({"path": ["L-", "L+", "R+", "R-", "L-"], "angles": [phi1, mu, mu, mu, phi2]})
    return type_list


def CC_CC_CC_paths_generation(r_min, Alpha, type_list):
    """
    Generate candidate paths of type CC|CC|CC.

    Parameters:
        r_min: scalar representing scaled minimum turning radius.
        Alpha: 3x3 rotation matrix in SO(3) representing the transformed terminal configuration.
        type_list: list of dictionary of current candidate paths, each dictionary with the sturcture {"path": [], "angles": []}.

    Returns:
        type_list: updated list of dictionary of current candidate paths
    """
    U_max = sqrt(1/r_min**2 - 1)
    # cosine of second, third, fourth, and fifth angle of the L+R+|R-L-|L+R+ and R-L-|L+R+|R-L- paths
    # Define the coefficients for the cubic equation in cos(mu)
    a4 = 32 * r_min**10 - 64 * r_min**8 + 56 * r_min**6 - 32 * r_min**4 + 10 * r_min**2 - 1 - Alpha[0, 0] * (r_min**2 - 1) - r_min * (Alpha[0, 2] * 
            sqrt(1 - r_min**2) - Alpha[2, 0] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min)
    a3 = -128 * r_min**10 + 288 * r_min**8 - 240 * r_min**6 + 104 * r_min**4 - 24 * r_min**2          
    a2 = 192 * r_min**10 - 480 * r_min**8 + 408 * r_min**6 - 136 * r_min**4 + 16 * r_min**2
    a1 = -128 * r_min**10 + 352 * r_min**8 - 320 * r_min**6 + 96 * r_min**4
    a0 = 32 * r_min**10 - 96 * r_min**8 + 96 * r_min**6 - 32 * r_min**4 

    # Solve the polynomial equation
    coefficients = [a0, a1, a2, a3, a4]
    # Solve the cubic equation
    roots = np.roots(coefficients)
    #print(roots)
    # Filter for real roots
    c_mu_sols = roots[np.isreal(roots)].real
    for c_mu in c_mu_sols:
        c_mu = cosine_value_clip(c_mu)
        if c_mu >= -1 and c_mu <= 1:  # if has solution
            mu_sol = [acos(c_mu), 2 * pi - acos(c_mu)]
            for mu in mu_sol:
                if mu < atan(1/sqrt(U_max**4-1)) + pi/2+ 0.00001:
                    A = 16 * (r_min**2 - 1)**2 * r_min * sin(mu / 2)**3 * cos(mu / 2) * (6 * r_min**4 + 2 * (r_min**2 - 1) * r_min**2 * 
                            cos(2 * mu) - 4 * r_min**2 + (6 * r_min**2 - 8 * r_min**4) * cos(mu) + 1)
                    B = 2 * (r_min**2 - 1) * r_min * (-16 * r_min**8 * cos(3 * mu) + 2 * r_min**8 * cos(4 * mu) + 70 * r_min**8 + 36 * r_min**6 * cos(3 * mu) - 
                        5 * r_min**6 * cos(4 * mu) - 135 * r_min**6 - 25 * r_min**4 * cos(3 * mu) + 4 * r_min**4 * cos(4 * mu) + 88 * r_min**4 + 5 * r_min**2 
                        * cos(3 * mu) - r_min**2 * cos(4 * mu) - 22 * r_min**2 + (-112 * r_min**8 + 220 * r_min**6 - 143 * r_min**4 + 35 * r_min**2 - 2) * cos(mu) 
                        + (56 * r_min**8 - 116 * r_min**6 + 76 * r_min**4 - 17 * r_min**2 + 1) * cos(2 * mu) + 2)                 
                    # cos(phi1-gamma)
                    c_phi1_minus_gamma = (140 * r_min**11 - 340 * r_min**9 + 296 * r_min**7 - 112 * r_min**5 + 18 * r_min**3 + Alpha[2, 0] * sqrt(1 - r_min**2) - 
                                          8 * (r_min**2 - 1)**2 * (4 * r_min**2 - 3) * r_min**5 * cos(3 * mu) + 4 * (r_min**2 - 1)**3 * r_min**5 * cos(4 * mu) - 
                                          8 * (28 * r_min**8 - 69 * r_min**6 + 60 * r_min**4 - 22 * r_min**2 + 3) * r_min**3 * cos(mu) + 4 * (28 * r_min**8 - 
                                          72 * r_min**6 + 63 * r_min**4 - 21 * r_min**2 + 2) * r_min**3 * cos(2 * mu) - Alpha[2, 2] * r_min - r_min) / max(sqrt(A**2 + B**2) , 10**(-8))
                    
                    c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)
                    
                    C = 16 * (r_min**2 - 1)**2 * r_min * sin(mu / 2)**3 * cos(mu / 2) * (6 * r_min**4 + 2 * (r_min**2 - 1) * r_min**2 * cos(2 * mu) - 4 * r_min**2 + 
                            (6 * r_min**2 - 8 * r_min**4) * cos(mu) + 1)
                    D = 2 * (r_min**2 - 1) * r_min * (-16 * r_min**8 * cos(3 * mu) + 2 * r_min**8 * cos(4 * mu) + 70 * r_min**8 + 36 * r_min**6 * cos(3 * mu) - 5 * r_min**6 * 
                            cos(4 * mu) - 135 * r_min**6 - 25 * r_min**4 * cos(3 * mu) + 4 * r_min**4 * cos(4 * mu) + 88 * r_min**4 + 5 * r_min**2 * cos(3 * mu) - r_min**2 * 
                            cos(4 * mu) - 22 * r_min**2 + (-112 * r_min**8 + 220 * r_min**6 - 143 * r_min**4 + 35 * r_min**2 - 2) * cos(mu) + (56 * r_min**8 - 116 * r_min**6 + 
                            76 * r_min**4 - 17 * r_min**2 + 1) * cos(2 * mu) + 2)
                    # cos(phi2-sigma)
                    c_phi2_minus_sigma = (140 * r_min**11 - 340 * r_min**9 + 296 * r_min**7 - 112 * r_min**5 + 18 * r_min**3 - Alpha[0,2] * sqrt(1 - r_min**2) - 
                                          8 * (r_min**2 - 1)**2 * r_min**5 * (4 * r_min**2 - 3) * cos(3 * mu) + 4 * (r_min**2 - 1)**3 * r_min**5 * cos(4 * mu) - 
                                          8 * (28 * r_min**8 - 69 * r_min**6 + 60 * r_min**4 - 22 * r_min**2 + 3) * r_min**3 * cos(mu) + 4 * (28 * r_min**8 - 
                                          72 * r_min**6 + 63 * r_min**4 - 21 * r_min**2 + 2) * r_min**3 * cos(2 * mu) - Alpha[2, 2] * r_min - r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
                    
                    c_phi2_minus_sigma = cosine_value_clip(c_phi2_minus_sigma)
                    
                    if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi2_minus_sigma >= -1 and c_phi2_minus_sigma <= 1:  # if have solution
                        phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                        phi2_minus_sigma_sol = [acos(c_phi2_minus_sigma), 2 * pi - acos(c_phi2_minus_sigma)]
                        
                        for enum1 in phi1_minus_gamma_sol:
                            # calculate phi1
                            phi1 = enum1 + atan2(A , B)
                            if abs(phi1-2*pi)<10**(-3):
                                phi1 = 0.0
                            if phi1 < -10**(-3):
                                phi1 += 2 * pi
                            elif phi1 > 2 * pi:
                                phi1 -= 2 * pi
                            for enum2 in phi2_minus_sigma_sol:
                                # calculate phi2
                                phi2 = enum2 + atan2(C , D)
                                if abs(phi2-2*pi)<10**(-3):
                                    phi2 = 0.0
                                if phi2 < -10**(-3):
                                    phi2 += 2 * pi
                                elif phi2 > 2 * pi:
                                    phi2 -= 2 * pi
                                # add candidate paths to list
                                type_list.append({"path": ["L+", "R+", "R-", "L-", "L+", "R+"], "angles": [phi1, mu, mu, mu, mu, phi2]})
                                type_list.append({"path": ["R-", "L-", "L+", "R+", "R-", "L-"], "angles": [phi1, mu, mu, mu, mu, phi2]})
                                

    # cosine of second, third, fourth, and fifth angle of the R+L+|L-R-|R+L+ and L-R-|R+L+|L-R- paths
    # Define the coefficients for the cubic equation in cos(mu)
    a4 = 32 * r_min**10 - 64 * r_min**8 + 56 * r_min**6 - 32 * r_min**4 + 10 * r_min**2 - 1 - Alpha[0, 0] * (r_min**2 - 1) - r_min * (-Alpha[0, 2] * 
            sqrt(1 - r_min**2) + Alpha[2, 0] * sqrt(1 - r_min**2) + Alpha[2, 2] * r_min)
    a3 = -128 * r_min**10 + 288 * r_min**8 - 240 * r_min**6 + 104 * r_min**4 - 24 * r_min**2          
    a2 = 192 * r_min**10 - 480 * r_min**8 + 408 * r_min**6 - 136 * r_min**4 + 16 * r_min**2
    a1 = -128 * r_min**10 + 352 * r_min**8 - 320 * r_min**6 + 96 * r_min**4
    a0 = 32 * r_min**10 - 96 * r_min**8 + 96 * r_min**6 - 32 * r_min**4 

    # Solve the polynomial equation
    coefficients = [a0, a1, a2, a3, a4]
    # Solve the cubic equation
    roots = np.roots(coefficients)
    # Filter for real roots
    c_mu_sols = roots[np.isreal(roots)].real

    for c_mu in c_mu_sols:
        c_mu = cosine_value_clip(c_mu)
        if c_mu >= -1 and c_mu <= 1:  # if has solution
            mu_sol = [acos(c_mu), 2 * pi - acos(c_mu)]
            for mu in mu_sol:
                if mu < atan(1/sqrt(U_max**4-1)) + pi/2+ 0.00001:
                    A = 16 * (r_min**2 - 1)**2 * r_min * sin(mu / 2)**3 * cos(mu / 2) * (6 * r_min**4 + 2 * (r_min**2 - 1) * r_min**2 * 
                            cos(2 * mu) - 4 * r_min**2 + (6 * r_min**2 - 8 * r_min**4) * cos(mu) + 1)
                    B = 2 * (r_min**2 - 1) * r_min * (-16 * r_min**8 * cos(3 * mu) + 2 * r_min**8 * cos(4 * mu) + 70 * r_min**8 + 36 * r_min**6 * cos(3 * mu) - 
                        5 * r_min**6 * cos(4 * mu) - 135 * r_min**6 - 25 * r_min**4 * cos(3 * mu) + 4 * r_min**4 * cos(4 * mu) + 88 * r_min**4 + 5 * r_min**2 
                        * cos(3 * mu) - r_min**2 * cos(4 * mu) - 22 * r_min**2 + (-112 * r_min**8 + 220 * r_min**6 - 143 * r_min**4 + 35 * r_min**2 - 2) * cos(mu) 
                        + (56 * r_min**8 - 116 * r_min**6 + 76 * r_min**4 - 17 * r_min**2 + 1) * cos(2 * mu) + 2)
                    # cos(phi1-gamma)
                    c_phi1_minus_gamma = (140 * r_min**11 - 340 * r_min**9 + 296 * r_min**7 - 112 * r_min**5 + 18 * r_min**3 - Alpha[2, 0] * sqrt(1 - r_min**2) - 
                                          8 * (r_min**2 - 1)**2 * (4 * r_min**2 - 3) * r_min**5 * cos(3 * mu) + 4 * (r_min**2 - 1)**3 * r_min**5 * cos(4 * mu) - 
                                          8 * (28 * r_min**8 - 69 * r_min**6 + 60 * r_min**4 - 22 * r_min**2 + 3) * r_min**3 * cos(mu) + 4 * (28 * r_min**8 - 
                                          72 * r_min**6 + 63 * r_min**4 - 21 * r_min**2 + 2) * r_min**3 * cos(2 * mu) - Alpha[2, 2] * r_min - r_min) / max(sqrt(A**2 + B**2) , 10**(-8))
                    c_phi1_minus_gamma = cosine_value_clip(c_phi1_minus_gamma)

                    C = 16 * (r_min**2 - 1)**2 * r_min * sin(mu / 2)**3 * cos(mu / 2) * (6 * r_min**4 + 2 * (r_min**2 - 1) * r_min**2 * cos(2 * mu) - 4 * r_min**2 + 
                            (6 * r_min**2 - 8 * r_min**4) * cos(mu) + 1)
                    D = 2 * (r_min**2 - 1) * r_min * (-16 * r_min**8 * cos(3 * mu) + 2 * r_min**8 * cos(4 * mu) + 70 * r_min**8 + 36 * r_min**6 * cos(3 * mu) - 5 * r_min**6 * 
                            cos(4 * mu) - 135 * r_min**6 - 25 * r_min**4 * cos(3 * mu) + 4 * r_min**4 * cos(4 * mu) + 88 * r_min**4 + 5 * r_min**2 * cos(3 * mu) - r_min**2 * 
                            cos(4 * mu) - 22 * r_min**2 + (-112 * r_min**8 + 220 * r_min**6 - 143 * r_min**4 + 35 * r_min**2 - 2) * cos(mu) + (56 * r_min**8 - 116 * r_min**6 + 
                            76 * r_min**4 - 17 * r_min**2 + 1) * cos(2 * mu) + 2)
                    # cos(phi2-sigma)
                    c_phi2_minus_sigma = (140 * r_min**11 - 340 * r_min**9 + 296 * r_min**7 - 112 * r_min**5 + 18 * r_min**3 + Alpha[0,2] * sqrt(1 - r_min**2) - 
                                          8 * (r_min**2 - 1)**2 * r_min**5 * (4 * r_min**2 - 3) * cos(3 * mu) + 4 * (r_min**2 - 1)**3 * r_min**5 * cos(4 * mu) - 
                                          8 * (28 * r_min**8 - 69 * r_min**6 + 60 * r_min**4 - 22 * r_min**2 + 3) * r_min**3 * cos(mu) + 4 * (28 * r_min**8 - 
                                          72 * r_min**6 + 63 * r_min**4 - 21 * r_min**2 + 2) * r_min**3 * cos(2 * mu) - Alpha[2, 2] * r_min - r_min) / max(sqrt(C**2 + D**2) , 10**(-8))
                    c_phi2_minus_sigma - cosine_value_clip(c_phi2_minus_sigma)

                    if c_phi1_minus_gamma >= -1 and c_phi1_minus_gamma <= 1 and c_phi2_minus_sigma >= -1 and c_phi2_minus_sigma <= 1:  # if have solution
                        phi1_minus_gamma_sol = [acos(c_phi1_minus_gamma), 2 * pi - acos(c_phi1_minus_gamma)]
                        phi2_minus_sigma_sol = [acos(c_phi2_minus_sigma), 2 * pi - acos(c_phi2_minus_sigma)]
                        
                        for enum1 in phi1_minus_gamma_sol:
                            # calculate phi1
                            phi1 = enum1 + atan2(A , B)
                            if abs(phi1-2*pi)<10**(-3):
                                phi1 = 0.0
                            if phi1 < -10**(-3):
                                phi1 += 2 * pi
                            elif phi1 > 2 * pi:
                                phi1 -= 2 * pi
                            
                            for enum2 in phi2_minus_sigma_sol:
                                # calculate phi2
                                phi2 = enum2 + atan2(C , D)
                                if abs(phi2-2*pi)<10**(-3):
                                    phi2 = 0.0
                                if phi2 < -10**(-3):
                                    phi2 += 2 * pi
                                elif phi2 > 2 * pi:
                                    phi2 -= 2 * pi
                                # add candidate paths to list
                                type_list.append({"path": ["R+", "L+", "L-", "R-", "R+", "L+"], "angles": [phi1, mu, mu, mu, mu, phi2]})
                                type_list.append({"path": ["L-", "R-", "R+", "L+", "L-", "R-"], "angles": [phi1, mu, mu, mu, mu, phi2]})
    return type_list

def path_generation_and_check(max_turn_rate, initial_config, terminal_config, sphere_radius=1):
    """
    Generate all paths that satisfy the angle restrictions in the sufficient list, 
    and check if they lead to the desired terminal configuration.

    Parameters:
        sphere_radius: scalar representing the radius of the sphere.
        max_turn_rate: scalar representing the maximum turning rate.
        initial_config: 3x3 rotation matrix in SO(3) representing the initial configuration.
        terminal_config: 3x3 rotation matrix in SO(3) representing the terminal configuration.

    Returns:
        feas_list: list of paths that lead to the desired terminal configuration.
    """
    # transform the problem to standard one
    U_max, r_min, Alpha = trans_2_standard(max_turn_rate, initial_config, terminal_config, sphere_radius)
    type_list = [] # list for generated paths
    feas_list = [] # list for checked feasible paths
    func_list = ["C", "G", "T", "CC", "GC", "C_C", "TC", "CC_C", "CGC", "C_CG", "CTC", "C_CC_C", "CGC_C", "CC_CC", "C_CGC_C","C_CC_CC","CC_CC_CC"]
    for path_type in func_list:
        function_name = f"{path_type}_paths_generation"
        # call all functions xxx_paths_generation() for xxx in path_list
        type_list = globals()[function_name](r_min, Alpha, type_list)
    # traverse the type_list
    for enum in type_list:
        trans_matrix_list = [] # list for transformation matrices of one path
        for seg, length in zip(enum["path"], enum["angles"]): # traverse each segment of a path
            trans_matrix_list.append(Rotation_cal(seg, length, U_max)) 
        
        # calculate the terminal configuration of the generated path
        path_term_config = np.eye(3)
        for trans_mat in trans_matrix_list:
            path_term_config = np.dot(path_term_config, trans_mat) 
        # check if the terminal configuration consists with the desired one
        feasible_flag = SO3_distance_check(path_term_config, Alpha)

        if feasible_flag:
            feas_list.append(enum) # append path to feasible list if checked feasible
    return  feas_list

def path_generation_and_check_sym(max_turn_rate, initial_config, terminal_config, sphere_radius=1):
    """
    Generate all paths that satisfy the angle restrictions and are in the symmetric forms (if applicable), 
    and check if they lead to the desired terminal configuration.

    Parameters:
        sphere_radius: scalar representing the radius of the sphere.
        max_turn_rate: scalar representing the maximum turning rate.
        initial_config: 3x3 rotation matrix in SO(3) representing the initial configuration.
        terminal_config: 3x3 rotation matrix in SO(3) representing the terminal configuration.

    Returns:
        feas_list: list of paths that lead to the desired terminal configuration.
        U_max: scalar representing the scaled maximum turning rate
    """
    # transform the problem to symmetric standard one (with terminal config as I_3)
    U_max, r_min, Alpha = trans_2_standard_sym(max_turn_rate, initial_config, terminal_config, sphere_radius)
    type_list = [] # list for generated paths
    feas_list = [] # list for checked feasible paths
    func_list = ["GC", "TC", "CC_C", "C_CG", "CGC_C", "C_CC_CC"]
    for path_type in func_list:
        function_name = f"{path_type}_paths_generation"
        # call all functions xxx_paths_generation() for xxx in path_list
        type_list = globals()[function_name](r_min, Alpha, type_list)
    
    # traverse the type_list
    for enum in type_list:
        trans_matrix_list = [] # list for transformation matrices of one path
        for seg, length in zip(enum["path"], enum["angles"]): # traverse each segment of a path
            trans_matrix_list.append(Rotation_cal(seg, length, U_max)) 
        
        # calculate the terminal configuration of the generated path
        path_term_config = np.eye(3)
        for trans_mat in trans_matrix_list:
            path_term_config = np.dot(path_term_config,trans_mat) 
        # check if the terminal configuration consists with the desired one
        feasible_flag = SO3_distance_check(path_term_config, Alpha)
        
        if feasible_flag:
            # transform the path to its symmetric form
            sym_path = []
            sym_lengths = []
            for i in range(len(enum["path"]) - 1, -1, -1):
                if enum["path"][i] == "L+":
                    sym_path.append("R-")
                elif enum["path"][i] == "R+":
                    sym_path.append("L-")
                elif enum["path"][i] == "G+":
                    sym_path.append("G-")
                elif enum["path"][i] == "L-":
                    sym_path.append("R+")
                elif enum["path"][i] == "R-":
                    sym_path.append("L+")
                elif enum["path"][i] == "G-":
                    sym_path.append("G+")
                sym_lengths.append(enum["angles"][i])
            feas_list.append({"path": sym_path, "angles": sym_lengths}) # append path to feasible list if checked feasible

    return  feas_list, U_max

def find_shortest_path(feas_list, r_min):
    """
    Finds the shortest path among a list of feasible paths.

    Parameters:
        feas_list: list of dictionaries of feasible paths
        r_min: scalar representing the scaled minimum turning radius
    Return:
        opti_path_dict: a dictionary of the optimal path
    """
    U_max = sqrt(1/r_min**2 - 1)
    opt_path = None
    opt_length_sum = float('inf')
    for enum in feas_list:
        temp_length = 0
        for p, l in zip(enum["path"], enum["angles"]):
            if p=="G+" or p=="G-":
                temp_length += l
            elif p=="L0" or p=="R0":
                temp_length += l / abs(U_max)
            else:
                temp_length += l * r_min
        if temp_length < opt_length_sum:
            opt_length_sum = temp_length
            opt_length = enum["angles"]
            opt_path = enum["path"]
    opt_path_dict = {"path": opt_path, "angles": opt_length}
    return opt_path_dict

def sample_seg_points(type, angle, N_points, ini_config_seg, U_max):
    """
    Samples points on a segment with specified type, angle in radians, number of sampled points, and initial configuration.

    Parameters:
        type: a string representing path type of the segment,
        angle: a scalar representing the angle of the segment in radians,
        N_points: a scalar representing the number of points sampled,
        ini_config_seg: a 3x3 matrix in SO(3) representing the starting configuration of the segment,
        U_max: a scalar representing the scaled maximum turning rate.

    Returns: 
        x_coord: a list of sampled x coordinates,
        y_coord: a list of sampled y coordinates,
        z_coord: a list of sampled z coordinates,
        term_config_seg: a 3x3 matrix in SO(3) representing the terminal configuration of the segment. 
    """
    x_coord = [ini_config_seg[0,0]]
    y_coord = [ini_config_seg[1,0]]
    z_coord = [ini_config_seg[2,0]]

    for i in range(N_points):
        angle_temp = angle / N_points * i
        R_temp = Rotation_cal(type, angle_temp, U_max) # calculate the transformation matrix of each sample point
        basis_temp = np.matmul(ini_config_seg, R_temp) # calculate the sample point
        x_coord.append(basis_temp[0, 0])
        y_coord.append(basis_temp[1, 0])
        z_coord.append(basis_temp[2, 0])
    R_full = Rotation_cal(type, angle, U_max)
    term_config_seg = np.matmul(ini_config_seg, R_full)
    return x_coord, y_coord, z_coord, term_config_seg

def sample_paths_points(feas_list, opt_path_dict, ini_config, U_max, sample_step=0.001):
    """
    Samples points on all feasible paths and the optimal path.

    Parameters:
        feas_list: a list of feasible paths,
        opt_path_dict: a dictionary of the optimal path,
        ini_config: a 3x3 matrix in SO(3) representing the initial configuration of the CRS vehicle,
        U_max: a scalar representing the scaled maximum turning rate,
        sample_step: the step in radians between each sample point.

    Returns:
        feas_list: the list of feasible paths with sampled points added,
        opt_path_dict: the dictionary of optimal path with sampled points added.
    """
    # traverse all feasible paths and take sample
    for enum in feas_list:
        ini_config_temp = ini_config
        path_x = []
        path_y = []
        path_z = []
        for i in range(len(enum["path"])):
            x_temp, y_temp, z_temp, ini_config_temp = sample_seg_points(enum["path"][i], enum["angles"][i], math.floor(enum["angles"][i]/sample_step), ini_config_temp, U_max)
            path_x.extend(x_temp)
            path_y.extend(y_temp)
            path_z.extend(z_temp)
        path_sample = np.array([path_x, path_y, path_z])
        enum["sample"] = path_sample
    
    # take sample of the optimal path
    ini_config_temp = ini_config
    path_x = []
    path_y = []
    path_z = []
    for i in range(len(opt_path_dict["path"])):
        x_temp, y_temp, z_temp, ini_config_temp = sample_seg_points(opt_path_dict["path"][i], opt_path_dict["angles"][i], math.floor(opt_path_dict["angles"][i]/sample_step), ini_config_temp, U_max)
        path_x.extend(x_temp)
        path_y.extend(y_temp)
        path_z.extend(z_temp)
    path_sample = np.array([path_x, path_y, path_z])
    opt_path_dict["sample"] = path_sample
    return feas_list, opt_path_dict

def clean_path_list(list):
    """
    Removes segments with zero length and remove duplicated paths and paths with segments longer than pi in a list.

    Paramters
        list: a list of paths.
    
    Returns
        unique_entries: cleaned list
    """
    # remove segments with zero length
    threshold = 1e-5
    for enum in list:
        # Use a list comprehension to filter both "path" and "angles"
        if len(enum["path"])==1:
            filtered = [(p, l) for p, l in zip(enum["path"], enum["angles"])]
        else:
            filtered = [(p, l) for p, l in zip(enum["path"], enum["angles"]) if l >= threshold]
        enum["path"] = [p for p, l in filtered]
        enum["angles"] = [l for p, l in filtered]
   
    # Remove paths with any segment longer than 
    list = [enum for enum in list if all(l <= np.pi + threshold for l in enum["angles"])]
    
    # remove duplicated paths
    unique_entries = []
    seen = {}
    for enum in list:
        path_key = tuple(enum["path"])
        current_lengths = np.array(enum["angles"])

        if path_key in seen:
            # We already have one or more angle arrays for this path_key
            already_seen_angles_list = seen[path_key]

            # Compare against each angles array we previously stored
            skip_this = False
            for old_angles in already_seen_angles_list:
                if np.allclose(old_angles, current_lengths, atol=threshold):
                    # It's "close enough" to a previously stored array
                    skip_this = True
                    break

            # If we found a "close enough" match, skip adding this one
            if skip_this:
                continue

            # Otherwise, append the new angles array to the list
            seen[path_key].append(current_lengths)
            unique_entries.append(enum)

        else:
            # First time we see this path_key
            # Initialize a new list of angles for this path
            seen[path_key] = [current_lengths]
            unique_entries.append(enum)
    return unique_entries

def generate_points_sphere(center, R):
    """
    This function generates points on a sphere whose center is given by the variable
    "center" and with a radius of R.

    Parameters
    ----------
    center : Numpy 1x3 array
        Contains the coordinates corresponding to the center of the sphere.
    R : Scalar
        Contains the radius of the sphere.

    Returns
    -------
    x_grid : Numpy nd array
        Contains the x-coordinate of the points on the sphere.
    y_grid : Numpy nd array
        Contains the y-coordinate of the points on the sphere.
    z_grid : Numpy nd array
        Contains the z-coordinate of the points on the sphere.

    """

    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(-np.pi / 2, np.pi / 2, 50)

    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Finding the coordinates of the points on the sphere in the global frame
    x_grid = center[0] + R * np.cos(theta_grid) * np.cos(phi_grid)
    y_grid = center[1] + R * np.sin(theta_grid) * np.cos(phi_grid)
    z_grid = center[2] + R * np.sin(phi_grid)

    return x_grid, y_grid, z_grid


def opt_plot_3D(x_coords_path, y_coords_path, z_coords_path, ini_config, term_config, filename="paths_sphere.html", path_legend=False,):
    """
    Plots the optimal path and the sphere.

    Parameter
        x_coords_path: list containing the x coordinates of the optimal path,
        y_coords_path: list containing the y coordinates of the optimal path,
        z_coords_path: list containing the z coordinates of the optimal path,
        ini_config: 3x3 matrix in SO(3) representing the initial configuration,
        term_config: 3x3 matrix in SO(3) representing the terminal configuration,
        filename: string of file name,
        path_legend: string of the legend of the optimal path.

    Returns
        fig_3D: 3D figure with the optimal path on sphere
    """
    fig_3D = plotting_functions()
    # Plotting the sphere
    fig_3D.surface_3D(
        generate_points_sphere(np.array([0, 0, 0]), 1)[0],
        generate_points_sphere(np.array([0, 0, 0]), 1)[1],
        generate_points_sphere(np.array([0, 0, 0]), 1)[2],
        "grey",
        None,
        0.7,
    )
    # Plotting the initial and final configurations
    fig_3D.points_3D(
        [ini_config[0, 0]],
        [ini_config[1, 0]],
        [ini_config[2, 0]],
        "black",
        "$\\Large\\text{Initial position}$",
        "circle",
    )

    fig_3D.points_3D(
        [term_config[0, 0]],
        [term_config[1, 0]],
        [term_config[2, 0]],
        "red",
        "$\\Large\\text{Terminal position}$",
        "circle",
    )

    # Adding initial and final tangent vectors
    fig_3D.arrows_3D(
        [ini_config[0, 0]],
        [ini_config[1, 0]],
        [ini_config[2, 0]],
        [ini_config[0, 1]],
        [ini_config[1, 1]],
        [ini_config[2, 1]],
        "orange",
        "oranges",
        "$\\Large\\mathbf{T_v}(0)$",
        3,
        0.8,
        0.4,
        "n",
    )
    
    fig_3D.arrows_3D(
        [term_config[0, 0]],
        [term_config[1, 0]],
        [term_config[2, 0]],
        [term_config[0, 1]],
        [term_config[1, 1]],
        [term_config[2, 1]],
        "magenta",
        "agsunset",
        "$\\Large\\mathbf{T_v}(T)$",
        3,
        0.8,
        0.4,
        "n",
    )

    fig_3D.update_layout_3D(
        "X (m)", "Y (m)", "Z (m)", "Initial and final configurations"
    )
    fig_3D.scatter_3D(
        x_coords_path,
        y_coords_path,
        z_coords_path,
        "blue",
        path_legend,
    )

    # Writing the figure on the html file
    fig_3D.writing_fig_to_html(filename, "w")

    return fig_3D


def opt_path_gen(max_turn_rate, initial_config, terminal_config, sphere_radius=1, sample_step=0.001):
    """
    Generate the optimal path, along with all feasible paths in the sufficient list connecting the desired initial and terminal configurations.

    Parameters:
        max_turn_rate: a scalar representing the maximum turning rate,
        initial_config: a 3x3 matrix in SO(3) representing the initial configuration of the CRS vehicle,
        terminal_config: a 3x3 matrix in SO(3) representing the terminal configuration of the CRS vehicle,
        sphere_radius (optional): a scalar representing the radius of the sphere considered.
        sample_step (optional): a scalar representing the step in radians between two sampled point.

    Returns:
        feas_list: a list of feasible paths containing segment types, segment lengths, and sampled points,
        opt_path_dict: a dictionary of the optimal path containing segment types, segment lengths, and sampled points.
    """
    # generate all paths in the sufficient list and check for terminal configuration
    feas_list = path_generation_and_check(max_turn_rate, initial_config, terminal_config, sphere_radius)
    
    # generate paths in the symmetric forms and check for terminal configuration
    feas_list_sym, U_max = path_generation_and_check_sym(max_turn_rate, initial_config, terminal_config, sphere_radius)
    feas_list.extend(feas_list_sym) # add two lists together
    
    # remove zero-length segments, duplicated paths, and paths with segments with length 2pi
    feas_list = clean_path_list(feas_list)
    opt_path_dict = find_shortest_path(feas_list, 1/sqrt(1+U_max**2)) # find the optimal path
    feas_list.remove(opt_path_dict) # remove optimal one from the list of feasible paths
    
    # sample points from all feasible paths and the optimal path
    feas_list, opt_path_dict = sample_paths_points(feas_list, opt_path_dict, initial_config,  U_max, sample_step)
    return feas_list, opt_path_dict


def CRS_plot(feas_list, opt_path_dict, ini_config, term_config):
    """
    Plots both the optimal path and feasible paths on sphere, if feasible paths not wanted, let feas_list = [].

    Parameters
        feas_list: list of feasible paths,
        opt_path_dict: dictionary of the optimal path,
        ini_config: 3x3 matrix in SO(3) representing the initial configuration,
        term_config: 3x3 matrix in SO(3) representing the terminal configuration.
    """
    # plot optimal path on sphere
    opt_legend = "$\\Large " + "".join(f"{s[0]}^{s[1]}" for s in opt_path_dict["path"]) + "\\text{ (Optimal)}$"
    fig = opt_plot_3D(opt_path_dict["sample"][0,:], opt_path_dict["sample"][1,:], opt_path_dict["sample"][2,:], ini_config, term_config, path_legend=opt_legend)
    
    # add other feasible paths onto the figure
    colormap = get_cmap("plasma")
    restricted_start = 0.0  
    restricted_end = 1    # Stay below the yellow-heavy part
    for i in range(len(feas_list)):
        feas_legend = "$\\Large " + "".join(f"{s[0]}^{s[1]}" for s in feas_list[i]["path"]) + "$"
        color_fraction = restricted_start + (restricted_end - restricted_start) * (i / len(feas_list))
        rgba_color = colormap(color_fraction)
        color = f"rgba({rgba_color[0] * 255:.0f}, {rgba_color[1] * 255:.0f}, {rgba_color[2] * 255:.0f}, 1)"
        fig.scatter_3D(feas_list[i]["sample"][0,:], feas_list[i]["sample"][1,:], feas_list[i]["sample"][2,:], color, feas_legend, linestyle="dash")
    
    fig.show()
