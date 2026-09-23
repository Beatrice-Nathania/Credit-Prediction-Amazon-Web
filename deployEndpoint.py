"""Deploy the trained sklearn Pipeline to a SageMaker real-time endpoint."""

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import json

# ---- EDIT THESE ---------------------------------------------------------
BUCKET = "bet-bucket-110"
MODEL_S3_KEY = "credit/model.tar.gz"
ENDPOINT_NAME = "data-endpoint-4"
# -------------------------------------------------------------------------

REGION = "us-east-1"
INSTANCE_TYPE = "ml.m5.xlarge"
FRAMEWORK_VERSION = "1.2-1"


def get_lab_role_arn() -> str:
    iam = boto3.client("iam")
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]


def main() -> None:
    boto3.setup_default_session(region_name=REGION)
    sm_session = sagemaker.Session()
    role_arn = get_lab_role_arn()
    model_s3_uri = f"s3://{BUCKET}/{MODEL_S3_KEY}"

    print(f"Role:      {role_arn}")
    print(f"Model URI: {model_s3_uri}")
    print(f"Endpoint:  {ENDPOINT_NAME}")

    model = SKLearnModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point="inference.py",
        source_dir="src",
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sm_session,
    )

    print("\nDeploying endpoint (5-8 minutes)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    # PERBAIKAN: Sampel data disesuaikan dengan 28 kolom fitur data_D.csv (tanpa target Credit_Score)
    sample = {
        "instances": [
            [
                0,                                        # Unnamed: 0
                "0x20c27",                                # ID
                "CUS_0xf64",                              # Customer_ID
                "June",                                   # Month
                "James Regang",                           # Name
                32,                                       # Age
                "478-73-8323",                            # SSN
                "Doctor",                                 # Occupation
                56125.5,                                  # Annual_Income
                4875.125,                                 # Monthly_Inhand_Salary
                8,                                        # Num_Bank_Accounts
                3,                                        # Num_Credit_Card
                18,                                       # Interest_Rate
                2,                                        # Num_of_Loan
                "Credit-Builder Loan, and Mortgage Loan", # Type_of_Loan
                30,                                       # Delay_from_due_date
                14,                                       # Num_of_Delayed_Payment
                17.89,                                    # Changed_Credit_Limit
                4.0,                                      # Num_Credit_Inquiries
                "Standard",                               # Credit_Mix
                370.22,                                   # Outstanding_Debt
                32.014181,                                # Credit_Utilization_Ratio
                "28 Years and 10 Months",                 # Credit_History_Age
                "Yes",                                    # Payment_of_Min_Amount
                81.822856,                                # Total_EMI_per_month
                182.065510,                               # Amount_invested_monthly
                "High_spent_Medium_value_payments",       # Payment_Behaviour
                473.624133                                # Monthly_Balance
            ]
        ]
    }

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(sample),
    )
    print("\nSmoke test response:")
    print(response["Body"].read().decode("utf-8"))

    print(
        f"\nEndpoint '{ENDPOINT_NAME}' is live in {REGION}.\n"
        f"Delete it before lab teardown: predictor.delete_endpoint()"
    )


if __name__ == "__main__":
    main()