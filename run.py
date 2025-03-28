import subprocess
import sys

# List of Python files to execute sequentially along with their config files and specific flags
python_files_with_configs_and_flags = [
    ("stage0.py", ["--config", "configs/cofig_crossmoda2021.yaml"]),
    ("stage1.py", ["--config", "configs/cofig_crossmoda2021.yaml"]),
#    ("script2.py", ["--settings", "config2.json"]),
#    ("script3.py", ["--parameters", "config3.json"])
]

def run_python_file(file_name, args):
    try:
        print(f"Running {file_name} with arguments {args}...")
        result = subprocess.run([sys.executable, file_name] + args, check=True, text=True)
        print(f"Finished running {file_name} with return code {result.returncode}.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {file_name} with arguments {args}: {e}")
        sys.exit(e.returncode)

def main():
    for file, args in python_files_with_configs_and_flags:
        run_python_file(file, args)

if __name__ == "__main__":
    main()
