import serial
# import numpy as np
import pandas as pd


# TODO: Definir como es la trama [:p q1 q2] o [:p q1 .. q2 ..]
# TODO: En base a lo anteriror modificar int2hex
# TODO: Test serial port
def int2hex(trajectory):
    tra_hex = ''
    for point in trajectory:
        tra_hex += f'{format(point, "04x")}' + ' '
    return tra_hex[:-1]


def read_csv(path):
    df = pd.read_csv(path)
    q = df[['q1', 'q2']].to_numpy()
    z = df['z'].to_numpy()
    time = df['timestamp'].to_numpy()
    return q, z, time


def main():
    ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200)
    print(ser.name)  # check which port was really used
    q, z, t = read_csv(
        "/home/jere/Documentos/five-bar-robot/src/motion_planning\
            /trajectories/trajectory.csv")
    list_encode = int2hex(q[:, 0])
    list_encode += '\n'
    print(list_encode)
    ser.write(list_encode.encode())  # write a string
    serialString = ser.read(10)
    # Print the contents of 0he serial data
    print(serialString.decode('Ascii'))

    ser.close()  # close port


if __name__ == '__main__':
    main()
