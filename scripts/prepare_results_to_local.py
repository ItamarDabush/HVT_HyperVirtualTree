import pandas as pd
import subprocess
import os
import datetime
import pexpect
import getpass

def save_metrics_to_excel(metrics_df, output_folder, output_filename):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the full path for the output file
    output_file = os.path.join(output_folder, output_filename)

    # Save the DataFrame to an Excel file
    metrics_df.to_excel(output_file, index=False)
    print(f"Excel file saved to {output_file}")

    return output_file

def prepare_output_to_local(metrics_df):
    now = datetime.datetime.now()
    formatted_date_time = now.strftime('%Y_%m_%d_%H_%M_%S')
    # Remote server details
    remote_user = 'gilb-server'
    remote_host = '0.tcp.ngrok.io'
    remote_port = 15873
    remote_folder = '/home/itamar/HyperDecisioNet/validation_results'
    remote_filename = f'model_evaluation_metrics_{formatted_date_time}.xlsx'
    remote_file_path = os.path.join(remote_folder, remote_filename)

    # Local machine details
    local_folder = '"C:/Users/Itamar-pc/Desktop/Sofware_Thesis/DecisioNet_Hyper_final/validation_output"'

    # Save the DataFrame to an Excel file on the remote server
    # Execute this step on the remote server
    save_metrics_to_excel(metrics_df, remote_folder, remote_filename)

    # Transfer the file to your local machine
    scp_command = f"scp -P {remote_port} {remote_user}@{remote_host}:{remote_file_path} {local_folder}"
    print(f'Run this command at local: {scp_command}')
def main():
    metrics_df = pd.DataFrame({
        'Scope': ['Entire Model', 'Branch_1'],
        'Class': ['All', 'All'],
        'Total Accuracy': [95.0, 90.0],
        'Sigma Accuracy': [94.0, 89.0]
    })
    prepare_output_to_local(metrics_df)

if __name__ == "__main__":
    main()