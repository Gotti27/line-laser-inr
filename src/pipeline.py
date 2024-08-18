import subprocess

target = 'bunny'
images_number = [20, 50, 100, 150, 200, 300, 400, 500, 600, 1000]

print('start')
for n in images_number:
    print(f"started {n}")
    command = ['python', 'src/3d_train.py', '-t', target, '-i', str(n), '-m', 'uniform', '-e', str(0.05)]
    with open(f'logs/{target}-uniform-{n}.log', 'w') as logfile:
        subprocess.call(command, stdout=logfile, shell=False)

    command = ['python', 'src/3d_train.py', '-t', target, '-i', str(n), '-m', 'gradient', '-e', str(0.05)]
    with open(f'logs/{target}-gradient-{n}.log', 'w') as logfile:
        subprocess.call(command, stdout=logfile, shell=False)

print("done")
