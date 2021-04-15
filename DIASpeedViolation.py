import pandas as pd
data = pd.read_csv('Dataset-1_+.csv')

def speedviolation(speed, latitude, longitude, thresholdspeed):
    TempSpeedViolation = []
    for i in speed.index:
        if float(speed[i]) > thresholdspeed:
            TempSpeedViolation.append([latitude[i], longitude[i], speed[i], i])
    speed_violation = pd.DataFrame(data=TempSpeedViolation, columns=['Latitude', 'Longitude', 'Speed', 'Index'])
    return speed_violation


A = speedviolation(data['Speed'], data['Latitude'], data['Longitude'], 20)
print(A)
