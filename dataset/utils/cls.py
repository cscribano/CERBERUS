# -*- coding: utf-8 -*-
# ---------------------

LANE_CLS = {
    'single yellow': 0,
    'single white': 1,
    'crosswalk': 2,
    'double white': 3,
    'double other': 4,
    'road curb': 5,
    'single other': 6,
    'double yellow': 7
}

DET_CLS = {
    'pedestrian' : 0,
    'rider' : 1,
    'car' : 2,
    'truck' : 3,
    'bus' : 4,
    'train' : 5,
    'motorcycle' : 6,
    'bicycle' : 7,
    'traffic light' : 8,
    'traffic sign' : 9,
    'other vehicle': 10,
    'other person': 11,
    'trailer': 12
}

OCL_VEHICLES = [2, 3, 4, 6] #<- not used!

"""
- weather: "rainy|snowy|clear|overcast|undefined|partly cloudy|foggy"
- scene: "tunnel|residential|parking lot|undefined|city street|gas stations|highway|"
- timeofday: "daytime|night|dawn/dusk|undefined"

"""

WTR_CLS = {
    "rainy": 0,
    "snowy": 1,
    "clear": 2,
    "overcast": 3,
    "partly cloudy": 4,
    "foggy": 5,
    "undefined": 6
}

SN_CLS = {
    "tunnel": 0,
    "residential": 1,
    "parking lot": 2,
    "city street": 3,
    "gas stations": 4,
    "highway": 5,
    "undefined": 6
}

TD_CLS = {
    "daytime": 0,
    "night": 1,
    "dawn/dusk": 2,
    "undefined": 3
}
