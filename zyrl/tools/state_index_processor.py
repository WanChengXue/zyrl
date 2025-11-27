import os
import numpy as np
import pandas as pd
from typing import Any


def generate_state_table_delta5():
    index_state_dict = {}
    state_index_mapping = {}
    index = 0
    prob_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for i in range(6):
        if i == 0:
            first_region = (0, 0)
        else:
            first_region = (prob_list[i - 1], prob_list[i])
        for j in range(6):
            if j == 0:
                second_region = (0, 0)
            else:
                second_region = (prob_list[j - 1], prob_list[j])
            for k in range(6):
                if k == 0:
                    third_region = (0, 0)
                else:
                    third_region = (prob_list[k - 1], prob_list[k])
                for l in range(6):
                    if l == 0:
                        fourth_region = (0, 0)
                    else:
                        fourth_region = (prob_list[l - 1], prob_list[l])
                    for m in range(6):
                        if m == 0:
                            fifth_region = (0, 0)
                        else:
                            fifth_region = (prob_list[m - 1], prob_list[m])
                        index_state_dict[index] = (
                            first_region,
                            second_region,
                            third_region,
                            fourth_region,
                            fifth_region,
                        )
                        state_index_mapping[
                            (
                                first_region,
                                second_region,
                                third_region,
                                fourth_region,
                                fifth_region,
                            )
                        ] = index
                        index += 1
    return index_state_dict, state_index_mapping


def generate_state_table_delta4_with_pred():
    index_state_dict = {}
    state_index_mapping = {}
    index = 0
    prob_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    region_dict = {
        0: (-100, -10),
        1: (-10, -8),
        2: (-8, -6),
        3: (-6, -4),
        4: (-4, -2),
        5: (-2, 0),
        6: (0, 2),
        7: (2, 4),
        8: (4, 6),
        9: (6, 8),
        10: (8, 10),
        11: (10, 100),
    }
    for i in range(6):
        if i == 0:
            first_region = (0, 0)
        else:
            first_region = (prob_list[i - 1], prob_list[i])
        for j in range(6):
            if j == 0:
                second_region = (0, 0)
            else:
                second_region = (prob_list[j - 1], prob_list[j])
            for k in range(6):
                if k == 0:
                    third_region = (0, 0)
                else:
                    third_region = (prob_list[k - 1], prob_list[k])
                for l in range(6):
                    if l == 0:
                        fourth_region = (0, 0)
                    else:
                        fourth_region = (prob_list[l - 1], prob_list[l])
                    for region_index, region_interval in region_dict.items():
                        index_state_dict[index] = (
                            first_region,
                            second_region,
                            third_region,
                            fourth_region,
                            region_interval,
                        )
                        state_index_mapping[
                            (
                                first_region,
                                second_region,
                                third_region,
                                fourth_region,
                                region_interval,
                            )
                        ] = index
                        index += 1
    return index_state_dict, state_index_mapping


def generate_state_table_delta4():
    index_state_dict = {}
    state_index_mapping = {}
    index = 0
    prob_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for i in range(6):
        if i == 0:
            first_region = (0, 0)
        else:
            first_region = (prob_list[i - 1], prob_list[i])
        for j in range(6):
            if j == 0:
                second_region = (0, 0)
            else:
                second_region = (prob_list[j - 1], prob_list[j])
            for k in range(6):
                if k == 0:
                    third_region = (0, 0)
                else:
                    third_region = (prob_list[k - 1], prob_list[k])
                for l in range(6):
                    if l == 0:
                        fourth_region = (0, 0)
                    else:
                        fourth_region = (prob_list[l - 1], prob_list[l])
                    index_state_dict[index] = (
                        first_region,
                        second_region,
                        third_region,
                        fourth_region,
                    )
                    state_index_mapping[
                        (first_region, second_region, third_region, fourth_region)
                    ] = index
                    index += 1
    return index_state_dict, state_index_mapping


def generate_state_table_delta3_with_pred():
    index_state_dict = {}
    state_index_mapping = {}
    index = 0
    prob_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    region_dict = {
        0: (-100, -10),
        1: (-10, -8),
        2: (-8, -6),
        3: (-6, -4),
        4: (-4, -2),
        5: (-2, 0),
        6: (0, 2),
        7: (2, 4),
        8: (4, 6),
        9: (6, 8),
        10: (8, 10),
        11: (10, 100),
    }
    for i in range(6):
        if i == 0:
            first_region = (0, 0)
        else:
            first_region = (prob_list[i - 1], prob_list[i])
        for j in range(6):
            if j == 0:
                second_region = (0, 0)
            else:
                second_region = (prob_list[j - 1], prob_list[j])
            for k in range(6):
                if k == 0:
                    third_region = (0, 0)
                else:
                    third_region = (prob_list[k - 1], prob_list[k])
                for region_index, region_interval in region_dict.items():
                    index_state_dict[index] = (
                        first_region,
                        second_region,
                        third_region,
                        region_interval,
                    )
                    state_index_mapping[
                        (first_region, second_region, third_region, region_interval)
                    ] = index
                    index += 1
    return index_state_dict, state_index_mapping


def generate_state_table_delta3():
    index_state_dict = {}
    state_index_mapping = {}
    index = 0
    prob_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for i in range(6):
        if i == 0:
            first_region = (0, 0)
        else:
            first_region = (prob_list[i - 1], prob_list[i])
        for j in range(6):
            if j == 0:
                second_region = (0, 0)
            else:
                second_region = (prob_list[j - 1], prob_list[j])
            for k in range(6):
                if k == 0:
                    third_region = (0, 0)
                else:
                    third_region = (prob_list[k - 1], prob_list[k])

                index_state_dict[index] = (first_region, second_region, third_region)
                state_index_mapping[(first_region, second_region, third_region)] = index
                index += 1
    return index_state_dict, state_index_mapping
