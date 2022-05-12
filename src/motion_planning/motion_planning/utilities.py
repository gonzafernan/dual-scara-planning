import time
import serial
import pandas as pd
import numpy as np
from os.path import dirname, abspath
from os import makedirs


def normalize_trajectory(trajectory, factor):
    return np.array(trajectory * factor, dtype=np.int16)


def export_trajectory(vector, path, final_time, name="trajectory.csv"):
    """ Export pose to a csv

    Args:
        pose(np.ndarray): nx3 with positions, velocities or acceleration

        path : path to a directory where the file will be saved

        name : name of the file, must contain the extension .csv. \
        Default trajectory.csv
    """
    n = max(vector.shape)
    time, dt = np.linspace(start=0, stop=final_time, num=n, retstep=True)
    if np.round(dt, decimals=2) != 0.01:
        print("Warning: dt time rounded is not 0.01")
    time = np.round(time, decimals=2)

    data = {
        'q1': vector[:, 0],
        'q2': vector[:, 1],
        'z': vector[:, 1],
        'timestamp': time
    }
    df = pd.DataFrame(data)
    makedirs(path, exist_ok=True)
    path += '/' + name
    df.to_csv(path_or_buf=path, index=False)


# TODO: Definir como es la trama [:p q1 q2] o [:p q1 .. q2 ..]
# TODO: En base a lo anteriror modificar int2hex
def encode_trajectory(points):
    trama_hex = []
    for point in points:
        q1 = int(point[0]).to_bytes(length=4, byteorder='big',
                                    signed=True).hex()
        q2 = int(point[1]).to_bytes(length=4, byteorder='big',
                                    signed=True).hex()
        trama = ':S,' + q1 + ',' + q2 + '\r\n'
        trama_hex.append(trama.encode('utf-8'))
    return trama_hex


def read_csv(file):
    df = pd.read_csv(file)
    q = df[['q1', 'q2']].to_numpy()
    z = df['z'].to_numpy()
    time = df['timestamp'].to_numpy()
    return q, z, time


def main():
    ser = serial.Serial(port='/dev/ttyUSB0',
                        baudrate=115200,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE)
    print(ser.name)  # check which port was really used

    trj_path = dirname(dirname(abspath(__file__))) + '/trajectories'
    trj_name = '/trajectory.csv'
    trajectory = trj_path + trj_name
    q, _, _ = read_csv(trajectory)
    list_encode = encode_trajectory(q)
    print(list_encode)

    ser.write(list_encode.encode())  # write a string
    time.sleep(1)
    serialString = ser.readlines()
    # Print the contents of 0he serial data
    print(serialString.decode('Ascii'))
    ser.close()  # close port


if __name__ == '__main__':
    main()
