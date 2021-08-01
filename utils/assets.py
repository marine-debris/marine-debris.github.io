# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: assets.py includes the appropriate mappings.
'''
import numpy as np

cat_mapping = { 'Marine Debris': 1,
                'Dense Sargassum': 2,
                'Sparse Sargassum': 3,
                'Natural Organic Material': 4,
                'Ship': 5,
                'Clouds': 6,
                'Marine Water': 7,
                'Sediment-Laden Water': 8,
                'Foam': 9,
                'Turbid Water': 10,
                'Shallow Water': 11,
                'Waves': 12,
                'Cloud Shadows': 13,
                'Wakes': 14,
                'Mixed Water': 15}

labels = ['Marine Debris','Dense Sargassum','Sparse Sargassum',
          'Natural Organic Material','Ship','Clouds','Marine Water','Sediment-Laden Water',
          'Foam','Turbid Water','Shallow Water','Waves','Cloud Shadows','Wakes',
          'Mixed Water']

roi_mapping = { '16PCC' : 'Motagua (16PCC)',
                '16PDC' : 'Ulua (16PDC)',
                '16PEC' : 'La Ceiba (16PEC)',
                '16QED' : 'Roatan (16QED)',
                '18QWF' : 'Haiti (18QWF)',
                '18QYF' : 'Haiti (18QYF)',
                '18QYG' : 'Haiti (18QYG)',
                '19QDA' : 'Santo Domingo (19QDA)',
                '30VWH' : 'Scotland (30VWH)',
                '36JUN' : 'Durban (36JUN)',
                '48MXU' : 'Jakarta (48MXU)',
                '48MYU' : 'Jakarta (48MYU)',
                '48PZC' : 'Danang (48PZC)',
                '50LLR' : 'Bali (50LLR)',
                '51RVQ' : 'Yangtze (51RVQ)',
                '52SDD' : 'Nakdong (52SDD)',
                '51PTS' : 'Manila (51PTS)'}

color_mapping ={'Marine Debris': 'red',
               'Dense Sargassum': 'green',
               'Sparse Sargassum': 'limegreen',
               'Marine Water': 'navy',
               'Foam': 'purple',
               'Clouds': 'silver',
               'Cloud Shadows': 'gray',
               'Natural Organic Material': 'brown',
               'Ship': 'orange',
               'Wakes': 'yellow', 
               'Shallow Water': 'darkturquoise', 
               'Turbid Water': 'darkkhaki', 
               'Sediment-Laden Water': 'gold', 
               'Waves': 'seashell',
               'Mixed Water': 'rosybrown'}

s2_mapping = {'nm440': 0,
              'nm490': 1,
              'nm560': 2,
              'nm665': 3,
              'nm705': 4,
              'nm740': 5,
              'nm783': 6,
              'nm842': 7,
              'nm865': 8,
              'nm1600': 9,
              'nm2200': 10,
              'Confidence': 11,
              'Class': 12}

indexes_mapping = {'NDVI': 0,
                   'FAI': 1,
                   'FDI': 2,
                   'SI': 3,
                   'NDWI': 4,
                   'NRD': 5,
                   'NDMI': 6,
                   'BSI': 7,
                   'Confidence': 8,
                   'Class': 9}

texture_mapping = {'CON': 0, 
                   'DIS': 1, 
                   'HOMO': 2, 
                   'ENER': 3, 
                   'COR': 4, 
                   'ASM': 5,
                   'Confidence': 6,
                   'Class': 7}

conf_mapping = {'High': 1,
                'Moderate': 2,
                'Low': 3}

report_mapping = {'Very close': 1,
                  'Away': 2,
                  'No': 3}

rf_features = ['nm440','nm490','nm560','nm665','nm705','nm740','nm783','nm842',
              'nm865','nm1600','nm2200','NDVI','FAI','FDI','SI','NDWI','NRD',
              'NDMI','BSI','CON','DIS','HOMO','ENER','COR','ASM']

def cat_map(x):
    return cat_mapping[x]

cat_mapping_vec = np.vectorize(cat_map)