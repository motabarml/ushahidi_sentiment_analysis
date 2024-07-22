import platform

current_system = platform.system()
run_locally = None

if current_system == 'Windows':
    run_locally = True
else:
    run_locally = False