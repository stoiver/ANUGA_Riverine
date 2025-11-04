# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import utm
import sys

# Define the path to scripts and data
workshop_dir = os.getcwd()
data_dir = os.path.join(workshop_dir, 'data')
model_inputs_dir = os.path.join(workshop_dir, 'model_inputs')
model_outputs_dir = os.path.join(workshop_dir, 'model_outputs')
if 'google.colab' in sys.modules:
    data_dir = os.path.join(data_dir, 'collab')
    model_inputs_dir = os.path.join(model_inputs_dir, 'collab')
    model_outputs_dir = os.path.join(model_outputs_dir, 'collab')
model_visuals_dir = os.path.join(workshop_dir, 'visuals')
model_validation_dir = os.path.join(workshop_dir, 'validation')

# Simulation time definition
sim_starttime = pd.to_datetime('2014-07-02 00:00:00', format="%Y-%m-%d %H:%M:%S", utc=True)
#sim_endtime   = sim_starttime + pd.to_timedelta(2, 'd')
sim_endtime = pd.to_datetime('2014-07-14 00:00:00', format="%Y-%m-%d %H:%M:%S", utc=True)
sim_timestep = pd.to_timedelta(1800, 's')
sim_total_duration = (sim_endtime-sim_starttime)
data_download_dates = (str(sim_starttime-pd.to_timedelta(1, 'd'))[0:10].replace(' ', '').replace(':', ''),
                       str(sim_endtime+pd.to_timedelta(1, 'd'))[0:10].replace(' ', '').replace(':', ''))
sim_time = np.arange(sim_starttime, sim_endtime+sim_timestep, sim_timestep)

sim_starttime_str = str(sim_starttime)[0:19].replace('-', '').replace(' ', '').replace(':', '')

# Output file naming
domain_name = 'Shellmouth_flood'
model_name = f'{sim_starttime_str}_{domain_name}_{sim_total_duration.days}_days'

# Boundary Conditions
discharge_gauge_x, discharge_gauge_y = utm.from_latlon(50.960278, -101.412222)[0:2] # Lake of Prairies
discharge_gauge_ID = ('05MD009', 'DAM', discharge_gauge_x, discharge_gauge_y)

level_gauge_x, level_gauge_y = utm.from_latlon(50.993889, -101.287222)[0:2] # Shellmouth bridge
level_gauge_ID = ('05MD801', 'Russel', level_gauge_x, level_gauge_y)

# Gauges filepath
f_discharge =  os.path.join(model_inputs_dir, 'Discharge_at_%s_%s-%s.csv' % (
               discharge_gauge_ID[1],
               data_download_dates[0].replace('-', ''), 
               data_download_dates[1].replace('-', '')))

f_level =  os.path.join(model_inputs_dir, 'Level_at_%s_%s-%s.csv' % (
              level_gauge_ID[1],
              data_download_dates[0].replace('-', ''), 
              data_download_dates[1].replace('-', '')))
