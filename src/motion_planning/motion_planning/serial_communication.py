import time
import serial
import pandas as pd
from os.path import dirname, abspath


# TODO: Definir como es la trama [:p q1 q2] o [:p q1 .. q2 ..]
# TODO: En base a lo anteriror modificar int2hex
def encode_trajectory(points):
    tra_hex = ''
    for point in points:
        q1 = f'{format(point[0], "04x")}'
        q2 = f'{format(point[1], "04x")}'
        tra_hex += ':P1,' + q1 + '\n' + ':P2,' + q2 + '\n'
    return tra_hex[:-1]


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
