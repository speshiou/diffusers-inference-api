def get_ksa_service_account():
    import requests

    # retrieve the specified service account of workload identity federation
    # https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity#verify_the_setup
    metadata_url = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/email"
    headers = {"Metadata-Flavor": "Google"}

    response = requests.get(metadata_url, headers=headers)

    if response.status_code == 200:
        email = response.text
        return email
    else:
        print("Failed to retrieve email")
        return None
