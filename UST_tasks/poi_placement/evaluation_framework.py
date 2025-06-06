import osmnx as ox
import numpy as np
from math import sin, cos, sqrt, atan2, radians

"""
Utility model and help functions.
"""

def prepare_graph(my_graph_file, my_node_file):
    """
    loads graph and nodes prepared in load_graph.py
    """
    my_graph = ox.load_graphml(my_graph_file)
    with open(my_node_file, "r") as file:
        my_node_list = eval(file.readline())
    return my_graph, my_node_list


def cost_single(my_node, my_station, my_node_dict, my_cost_dict):
    """
    calculate the social cost for one station
    """
    if my_node[0] == 8519485058 or my_node[0] == 6031305212:
        pass
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    # check if distance has to be calculated
    if s_pos[0] in my_node_dict[my_node[0]]:
        distance = my_node_dict[my_node[0]][s_pos[0]]
    else:
        distance = haversine(s_pos, my_node)
        my_node_dict[my_node[0]][s_pos[0]] = distance
    # check if cost has to be calculated
    try:
        _a = my_cost_dict[my_node[0]]
    except KeyError:
        my_cost_dict[my_node[0]] = {}
    # if s_pos[0] in my_cost_dict[my_node[0]]:
    #     cost_node = my_cost_dict[my_node[0]][s_pos[0]]
    # else:
    cost_travel = alpha * distance / VELOCITY
    cost_boring = (1 - alpha) / (distance if distance >= 1.0 else 1.0) * (s_dict["W_s"] + 1 / s_dict["service rate"])
    cost_node = weak_demand(my_node) * (cost_travel + cost_boring)
    my_cost_dict[my_node[0]][s_pos[0]] = cost_node

    return cost_node, my_node_dict, my_cost_dict


def station_seeking(my_plan, my_node_list, my_node_dict, my_cost_dict):
    """
    output station assignment: Each node gets assigned the charging station with minimal social cost
    """
    for the_node in my_node_list:
        if the_node[0] == 8519485058 or the_node[0] == 6031305212:
            pass
        cost_list = [cost_single(the_node, my_station, my_node_dict, my_cost_dict) for my_station in my_plan]
        costminindex = cost_list.index(min(cost_list))
        chosen_station = my_plan[costminindex]
        s_pos = chosen_station[0]
        the_node[1]["charging station"] = s_pos[0]
        the_node[1]["distance"] = my_node_dict[the_node[0]][s_pos[0]]
    return my_node_list, my_node_dict, my_cost_dict


################################################################################################
def installment_fee(my_station):
    """
    returns cost to install the respective chargers at that position
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    charger_cost = np.sum(INSTALL_FEE * s_x)
    fee = charger_cost + s_pos[1]['estate price']  # * price_parkingplace
    s_dict["fee"] = fee  # [fee] = €
    return my_station


def charging_capability(my_station):
    """
    returns the summed up charging capability of the CS
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = np.sum(CHARGING_POWER * s_x)
    s_dict["capability"] = total_capacity  # [capability] = kw
    return my_station


def weak_demand(my_node):
    demand = my_node[1]["demand"] * (1 - 0.1 * my_node[1]["private CS"])
    return demand

def weak_demand_num(my_node):
    demand_num = my_node[1]["demand_num"] - my_node[1]["private CS num"]
    return demand_num

def influence_radius(my_station):
    """
    gives the radius of the nodes whose charging demand the CS could satisfy
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = s_dict["capability"]
    radius_s = RADIUS_MAX * 1 / (1 + np.exp(-total_capacity / (100 * capacity_unit)))
    s_dict["radius"] = radius_s  # [radius] = m
    return my_station


def haversine(s_pos, my_node):
    """
    yields the approximate distance of two GPS points, middle computational cost
    """
    lon1, lat1 = s_pos[1]['x'], s_pos[1]['y']
    R_earth = 6372800  # approximate radius of earth. [R_earth] = m
    lon2, lat2 = my_node[1]['x'], my_node[1]['y']
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R_earth * c  # [distance] = m
    if distance < 0.1:  # to avoid ZeroDivisionError
        distance = 0.1
    return distance


def nodes_covered(my_station, my_node_list):
    """
    yields the number of nodes within the influence radius of the station
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    radius_s = s_dict["radius"]
    I_1 = sum([1 if haversine(s_pos, my_node) <= radius_s else 0 for my_node in my_node_list])
    return I_1


def node_coverage(my_plan, my_node):
    """
    yields the number of nodes within the influence radius of the station
    """
    I_1, I_2 = 0, 0
    priv_CS = my_node[1]["private CS"]
    for my_station in my_plan:
        s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
        radius_s = s_dict["radius"]
        distance = haversine(s_pos, my_node)
        if distance <= radius_s:
            I_1 += 1
    for ith in range(I_1):
        I_2 += 1 / (ith + 1)
    single_benefit = I_2 * (1 - 0.1 * priv_CS)
    return single_benefit


def node_coverage_fixed(my_plan, my_node):
    """
    yields the number of nodes within the influence radius of the station
    """
    cov = 0
    priv_CS = my_node[1]["private CS"]
    for my_station in my_plan:
        s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
        radius_s = s_dict["radius"]
        distance = haversine(s_pos, my_node)
        if distance <= radius_s:
            cov += 1

    cov_weaken = 0.0
    for ith in range(cov):
        cov_weaken += 1 / (ith + 1)

    cov_benefit = cov_weaken * (1 - priv_CS)
    return cov_benefit

def node_coverage_fixed_collection(my_plan, my_node):
    """
    yields the number of nodes within the influence radius of the station
    """
    cov = 0
    priv_CS = my_node[1]["private CS"]
    for my_station in my_plan:
        s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
        radius_s = s_dict["radius"]
        distance = haversine(s_pos, my_node)
        if distance <= radius_s:
            cov += 1

    cov_benefit = cov * (1 - priv_CS)
    return cov_benefit


def total_number_EVs(my_station, my_node_list):
    """
    yields total number of EVs coming to S in a unit time interval for charging
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    D_s = sum([1 / my_node[1]["distance"] * weak_demand(my_node) if my_node[1]["charging station"] == s_pos[0]
               else 0 for my_node in my_node_list])

    s_dict["D_s"] = D_s  # dimensionless
    return my_station


def get_total_number_EVs(my_station, my_node_list):
    """
    yields total number of EVs coming to S in a unit time interval for charging
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    D_s = sum([1 / my_node[1]["distance"] * ((1 - my_node[1]['private CS']) * my_node[1]['demand_num']) if my_node[1]["charging station"] == s_pos[0]
               else 0 for my_node in my_node_list])

    return D_s


def service_rate(my_station):
    """
    returns how many cars can be served within one hour
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    s_dict["service rate"] = s_dict["capability"] / BATTERY  # [service rate] = 1/h
    return my_station


def W_s(my_station):
    """
    returns the expected value of waiting time
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    tau_s = 1 / s_dict["service rate"]
    rho_s = s_dict["D_s"] * tau_s * time_unit  # dimensionless (shortened away)
    if rho_s >= 1:
        my_W_s = my_inf
        s_dict["W_s"] = my_W_s
    else:
        my_W_s = rho_s * tau_s / (2 * (1 - rho_s))  # W_s = expected waiting time at S, [W_s] = h
        s_dict["W_s"] = my_W_s
    return my_station

def get_expected_waiting_time(my_station, node_list):
    """
    returns the expected value of waiting time
    """
    # s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    # tau_s = 1 / s_dict["service rate"]
    # rho_s = get_total_number_EVs(my_station, node_list) * tau_s * time_unit  # dimensionless (shortened away)
    # my_W_s = rho_s * tau_s / (2 * (1 - rho_s))  # W_s = expected waiting time at S, [W_s] = h

    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    tau_s = 1 / s_dict["service rate"]
    rho_s = s_dict["D_s"] * tau_s * time_unit  # dimensionless (shortened away)
    if rho_s >= 1:
        my_W_s = my_inf
    else:
        my_W_s = rho_s * tau_s / (2 * (1 - rho_s))  # W_s = expected waiting time at S, [W_s] = h

    return my_W_s


def s_dictionary(my_station, my_node_list):
    """
    returns the dictionary for the station
    """
    my_station = installment_fee(my_station)
    my_station = charging_capability(my_station)
    my_station = influence_radius(my_station)
    my_station = total_number_EVs(my_station, my_node_list)
    my_station = service_rate(my_station)
    my_station = W_s(my_station)
    return my_station

def search_station_by_id(station_id, plan):
    for station in plan:
        if station[0][0] == station_id:
            return station

    return None

# SCORE #####################################################################
def social_benefit(my_plan, my_node_list):
    """
    returns the social benefit of the charging plan (our definition of benefit)
    """
    my_benefit = 0
    for my_node in my_node_list:
        I3 = node_coverage(my_plan, my_node)
        my_benefit += I3
    my_benefit /= len(my_node_list)
    return my_benefit


def social_benefit_fixed(my_plan, my_node_list):
    """
    returns the social benefit of the charging plan (our definition of benefit)
    """
    my_benefit = 0
    for my_node in my_node_list:
        cov_benefit = node_coverage_fixed(my_plan, my_node)
        my_benefit += cov_benefit
    my_benefit /= len(my_node_list)
    return my_benefit


def social_benefit_new_station_with_demand_collection(chosen_node, old_node_list, old_plan, new_plan):
    """
    returns the social benefit of the charging plan (our definition of benefit)
    """
    nearby_cov_benefit = 0.0
    nearby_cov_benefit_num = 0.0
    nearby_demand_benefit = 0.0
    nearby_distance = []

    nearby_node_ids = []
    nearby_nodes = []
    for i, node in enumerate(old_node_list):
        distance = haversine(chosen_node, node)
        if distance <= RADIUS_MAX:
            nearby_node_ids.append(i)
            nearby_nodes.append(node)

        cov_benefit = node_coverage_fixed(new_plan, node)
        cov_benefit_num = node_coverage_fixed_collection(new_plan, node)
        nearby_cov_benefit += cov_benefit
        nearby_cov_benefit_num += cov_benefit_num
    nearby_cov_benefit /= len(old_node_list)
    nearby_cov_benefit_num /= len(old_node_list)

    nearby_stations = []
    for i, station in enumerate(old_plan):
        distance = haversine(chosen_node, station[0])
        if distance <= RADIUS_MAX:
            nearby_stations.append(station)

    for i, node_id in enumerate(nearby_node_ids):
        # before placement
        old_node_info = old_node_list[node_id]
        station = search_station_by_id(old_node_info[1]['charging station'], old_plan)
        station_info_loc = station[0]

        distance_to_station = haversine(old_node_info, station_info_loc)
        weaken_demand_power = ((old_node_info[1]['demand_num'] - old_node_info[1]['private CS number']) * BATTERY) * (distance_to_station / 1000)
        nearby_demand_benefit += weaken_demand_power
        nearby_distance.append(distance_to_station)

    nearby_distance = np.mean(nearby_distance)

    nearby_waiting_time = get_waiting_time(nearby_stations, old_node_list) if nearby_stations else -1
    nearby_charging_time = get_charging_time(nearby_stations, old_node_list) if nearby_stations else -1

    return nearby_cov_benefit, nearby_cov_benefit_num, nearby_demand_benefit, nearby_distance, nearby_waiting_time, nearby_charging_time


def node_social_benefit(chosen_node, node_list, plan):
    """
    returns the social benefit of the charging plan (our definition of benefit)
    """
    nearby_cov_benefit = 0.0
    nearby_cov_benefit_num = 0.0
    nearby_demand_benefit = 0.0
    nearby_distance = []

    nearby_node_ids = []
    nearby_nodes = []
    for i, node in enumerate(node_list):
        distance = haversine(chosen_node, node)
        if distance <= RADIUS_MAX:
            nearby_node_ids.append(i)
            nearby_nodes.append(node)

        cov_benefit = node_coverage_fixed(plan, node)
        cov_benefit_num = node_coverage_fixed_collection(plan, node)
        nearby_cov_benefit += cov_benefit
        nearby_cov_benefit_num += cov_benefit_num
    nearby_cov_benefit /= len(node_list)
    nearby_cov_benefit_num /= len(node_list)

    nearby_stations = []
    for i, station in enumerate(plan):
        distance = haversine(chosen_node, station[0])
        if distance <= RADIUS_MAX:
            nearby_stations.append(station)

    for i, node_id in enumerate(nearby_node_ids):
        # before placement
        old_node_info = node_list[node_id]
        station = search_station_by_id(old_node_info[1]['charging station'], plan)
        station_info_loc = station[0]

        distance_to_station = haversine(old_node_info, station_info_loc)
        weaken_demand_power = (((old_node_info[1]['demand_num'] - old_node_info[1]['private CS number']) * BATTERY) *
                               (distance_to_station / 1000))
        nearby_demand_benefit += weaken_demand_power
        nearby_distance.append(distance_to_station)

    nearby_distance = np.mean(nearby_distance)

    nearby_waiting_time = get_waiting_time(nearby_stations, node_list) if nearby_stations else -1
    nearby_charging_time = get_charging_time(nearby_stations, node_list) if nearby_stations else -1

    return nearby_cov_benefit, nearby_cov_benefit_num, nearby_demand_benefit, nearby_distance, nearby_waiting_time, nearby_charging_time


def social_benefit_new_station_with_demand(chosen_node, old_node_list, old_plan, new_plan):
    """
    returns the social benefit of the charging plan (our definition of benefit)
    """
    nearby_cov_benefit = 0.0
    nearby_demand_benefit = 0.0
    nearby_distance = []

    nearby_nodes = []
    for i, node in enumerate(old_node_list):
        distance = haversine(chosen_node, node)
        if distance <= RADIUS_MAX:
            nearby_nodes.append(i)

        cov_benefit = node_coverage_fixed(new_plan, node)
        nearby_cov_benefit += cov_benefit
    nearby_cov_benefit /= len(old_node_list)

    for node_id in nearby_nodes:
        # before placement
        old_node_info = old_node_list[node_id]
        station = search_station_by_id(old_node_info[1]['charging station'], old_plan)
        station_info_loc = station[0]

        distance_to_station = haversine(old_node_info, station_info_loc)
        weaken_demand_power = ((old_node_info[1]['demand_num'] - old_node_info[1]['private CS number']) * BATTERY) * (distance_to_station / 1000)
        nearby_demand_benefit += weaken_demand_power
        nearby_distance.append(distance_to_station)

    nearby_distance = np.mean(nearby_distance)

    return nearby_cov_benefit, nearby_demand_benefit, nearby_distance


def travel_cost(my_node_list):
    """ yields the estimated travel time of all vehicles """
    my_cost_travel = sum([my_node[1]["distance"] * weak_demand(my_node) / VELOCITY for my_node in my_node_list])
    return my_cost_travel


def travel_time(my_node_list, speed=6.38):
    """ yields the estimated travel time of all vehicles """
    my_cost_travel = sum([my_node[1]["distance"] / speed for my_node in my_node_list])
    return my_cost_travel

def charging_time(my_plan):
    """
    yields the total charging time given the capability of the CS of the charging plan
    """
    my_charge_time = [my_station[2]["D_s"] / my_station[2]["service rate"] for my_station in my_plan]
    my_charge_time = sum(my_charge_time)
    return my_charge_time / time_unit


def waiting_time(my_plan):
    """
    returns the average total waiting time of the charging plan
    """
    my_wait_time = [my_station[2]["D_s"] * my_station[2]["W_s"] for my_station in my_plan]
    my_wait_time = sum(my_wait_time)
    return my_wait_time / time_unit

def get_charging_time(my_plan, node_list):
    """
    yields the total charging time given the capability of the CS of the charging plan
    """
    my_charge_time = [get_total_number_EVs(my_station, node_list) / my_station[2]["service rate"] for my_station in my_plan]
    my_charge_time = sum(my_charge_time)

    return my_charge_time


def get_waiting_time(my_plan, node_list):
    """
    returns the average total waiting time of the charging plan
    """
    my_wait_time = [get_total_number_EVs(my_station, node_list) * get_expected_waiting_time(my_station, node_list) for my_station in my_plan]
    my_wait_time = sum(my_wait_time)

    return my_wait_time

def social_cost(my_plan, my_node_list):
    """
    returns the social cost, i.e. the negative side of the charging plan
    """
    cost_travel = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_plan)  # dimensionless
    wait_time = waiting_time(my_plan)  # dimensionless
    cost_boring = charg_time + wait_time  # dimensionless
    my_social_cost = alpha * cost_travel + (1 - alpha) * cost_boring
    return my_social_cost


def existing_score(my_existing_plan, my_node_list):
    """
    computes the score of the existing infrastructure
    """
    my_benefit = social_benefit(my_existing_plan, my_node_list)
    travel_time = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_existing_plan)  # dimensionless
    wait_time = waiting_time(my_existing_plan)
    cost_boring = charg_time + wait_time  # dimensionless
    my_cost = alpha * travel_time + (1 - alpha) * cost_boring
    return my_benefit, my_cost, charg_time, wait_time, travel_time

def existing_score_fixed(my_existing_plan, my_node_list):
    """
    computes the score of the existing infrastructure
    """
    my_benefit = social_benefit_fixed(my_existing_plan, my_node_list)
    travel_time = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_existing_plan)  # dimensionless
    wait_time = waiting_time(my_existing_plan)
    cost_boring = charg_time + wait_time  # dimensionless
    my_cost = travel_time + cost_boring
    return my_benefit, my_cost, charg_time, wait_time, travel_time


def norm_score(my_plan, my_node_list, norm_benefit, norm_charg, norm_wait, norm_travel):
    """
    same as score, but normalised.
    """
    my_score = -my_inf
    if not my_plan:
        return my_score
    benefit = social_benefit(my_plan, my_node_list) / norm_benefit
    cost_travel = travel_cost(my_node_list) / norm_travel  # dimensionless
    charg_time = charging_time(my_plan) / norm_charg  # dimensionless
    wait_time = waiting_time(my_plan) / norm_wait  # dimensionless
    # cost = (alpha * cost_travel + (1 - alpha) * (charg_time + wait_time)) / 3
    cost = cost_travel + charg_time + wait_time
    # my_score = my_lambda * benefit - (1 - my_lambda) * cost
    my_score = benefit - cost

    return my_score, benefit, cost, charg_time, wait_time, cost_travel


def score(my_plan, my_node_list):
    """
    returns the final result, i.e., the social score
    """
    my_score = -my_inf
    benefit = 0
    cost = 0
    if not my_plan:
        return my_score, benefit, cost
    benefit = social_benefit(my_plan, my_node_list)  # dimensionless
    cost = social_cost(my_plan, my_node_list)
    my_score = my_lambda * benefit - (1 - my_lambda) * cost
    return my_score, benefit, cost


# Constraints checks ############################################################################
def station_capacity_check(my_plan):
    """
    check if number of stations exceed capacity
    """
    for my_station in my_plan:
        s_x = my_station[1]
        if sum(s_x) > K:
            print("Error: More chargers at the station than admitted: {} chargers".format(sum(s_x)))


def installment_cost_check(my_plan, my_basic_cost):
    """
    check if instalment costs exceed budget
    """
    total_inst_cost = sum([my_station[2]["fee"] for my_station in my_plan]) - my_basic_cost
    if total_inst_cost > BUDGET:
        print("Error: Maximal BUDGET for installation costs exceeded.")


def control_charg_decision(my_plan, my_node_list):
    for my_node in my_node_list:
        station_sum = sum([1 for my_station in my_plan if my_node[1]["charging station"] == my_station[0]])
        if station_sum > 1:
            print("Error: More than one station is assigned to a node.")


def waiting_time_check(my_plan):
    """
    check that wiating time is bounded
    """
    for my_station in my_plan:
        s_dict = my_station[2]
        if s_dict["W_s"] == my_inf:
            print("Error: Waiting time goes to infinity.")


def constraint_check(my_plan, my_node_list, basic_cost):
    """
    test if solution satisfies all constraints
    """
    installment_cost_check(my_plan, basic_cost)
    control_charg_decision(my_plan, my_node_list)
    station_capacity_check(my_plan)
    waiting_time_check(my_plan)


# Parameters ########################################################
alpha = 0.4
my_lambda = 1 / 3

K = 20  # maximal number of chargers at a station
RADIUS_MAX = 1000  # [radius_max] = m
INSTALL_FEE = np.array([2500, 3000, 3200])  # fee per installing a charger of type 1, 2 or 3. [fee] = ¥
CHARGING_POWER = np.array([7, 11, 21])  # [power] = kW, rounded
BATTERY = 60  # battery capacity, [BATTERY] = kWh

BUDGET = 20000000  # 20,000,000 ¥
price_parkingplace = 2000  # ¥

time_unit = 1  # [time_unit] = h, introduced for getting the units correctly
capacity_unit = 1  # [cap_unit] = kW, introduced for getting the units correctly
VELOCITY = 23 * 1000  # based on m per hour, but here dimensionless

my_inf = 10 ** 6
my_dis_inf = 10 ** 7
