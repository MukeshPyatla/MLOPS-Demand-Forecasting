# End-to-End MLOps: Intelligent Retail Demand Forecasting

This repository contains an end-to-end MLOps pipeline for a common business problem: forecasting future product demand. The entire infrastructure is provisioned on **Microsoft Azure** using Terraform, and the pipeline is automated with **GitHub Actions**.

## 1. Business Problem

Retail businesses constantly struggle with inventory management.
- **Overstocking**: Leads to high storage costs, wasted capital, and potential spoilage.
- **Stockouts**: Result in lost sales, poor customer satisfaction, and damaged brand reputation.

This project builds an automated ML system to provide accurate, timely demand forecasts, enabling data-driven inventory decisions and optimizing the supply chain.

## 2. MLOps Solution & Architecture

We implement a batch-processing MLOps pipeline on Azure that automatically retrains and generates forecasts on a schedule.

**Architecture Diagram:**

* **(ACTION)**: Create an architecture diagram and embed it here. Tools like diagrams.net (draw.io) are great for this. Your diagram should show:
  * GitHub Actions triggering the workflow.
  * Terraform provisioning resources.
  * Raw data (CSV) in Blob Storage.
  * Azure Databricks running the ETL/feature engineering job.
  * Processed data (Parquet) in Blob Storage.
  * Azure Machine Learning orchestrating the training job.
  * The training job using a compute cluster to train a Prophet model.
  * MLflow tracking experiments and the model being registered in the AML Model Registry.

![Architecture Diagram]
graph TD
    subgraph "CI/CD & Version Control"
        G[<img src='https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg' width='40' /> <br> GitHub Repo]
        GA[<img src='https://raw.githubusercontent.com/devicons/devicon/master/icons/githubactions/githubactions-original.svg' width='40' /> <br> GitHub Actions]
    end

    subgraph "Azure Cloud Platform"
        T[<img src='https://raw.githubusercontent.com/devicons/devicon/master/icons/terraform/terraform-original.svg' width='40' /> <br> Terraform]

        subgraph "Data Storage (Azure Blob)"
            BS_Raw[raw-data container]
            BS_Proc[processed-data container]
        end

        subgraph "ML Execution & Orchestration"
            AML[<img src='https://raw.githubusercontent.com/devicons/devicon/master/icons/azure/azure-original.svg' width='40' /> <br> AML Workspace]
            DB[<img src='https://raw.githubusercontent.com/devicons/devicon/master/icons/azuredatabricks/azuredatabricks-original.svg' width='40' /> <br> Azure Databricks ETL]
            Compute[AML Compute Cluster]
            Registry[AML Model Registry]
        end
    end

    %% Define the workflow connections
    G -- "1. Code Push" --> GA
    GA -- "2. Runs terraform apply" --> T
    T -- "3. Provisions Resources" --> AML & DB & BS_Raw & BS_Proc

    GA -- "4. Uploads CSV" --> BS_Raw
    GA -- "5. Triggers AML Pipeline" --> AML

    AML -- "6. Runs ETL Job on" --> DB
    DB -- "7. Reads Raw Data" --> BS_Raw
    DB -- "8. Writes Processed Data" --> BS_Proc

    AML -- "9. Runs Training Job on" --> Compute
    Compute -- "10. Reads Processed Data" --> BS_Proc
    Compute -- "11. Registers Model" --> Registry
    
    %% Style for modern look
    style G fill:#f9f9f9,stroke:#333,stroke-width:2px
    style GA fill:#f9f9f9,stroke:#333,stroke-width:2px
    style T fill:#f9f9f9,stroke:#333,stroke-width:2px
    style AML fill:#e0f7fa,stroke:#0078d4,stroke-width:2px
    style DB fill:#e0f7fa,stroke:#0078d4,stroke-width:2px
    style Compute fill:#e0f7fa,stroke:#0078d4,stroke-width:2px
    style Registry fill:#e0f7fa,stroke:#0078d4,stroke-width:2px
    style BS_Raw fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style BS_Proc fill:#fff3e0,stroke:#ff9800,stroke-width:2px

## 3. Tech Stack

- **Cloud**: Microsoft Azure
- **Infrastructure as Code (IaC)**: Terraform
- **CI/CD Automation**: GitHub Actions
- **Data Storage**: Azure Blob Storage (Azure Data Lake Storage Gen2)
- **Data Processing**: Azure Databricks (using PySpark)
- **ML Orchestration & Training**: Azure Machine Learning (Workspaces, Compute, Pipelines, Model Registry)
- **ML Experiment Tracking**: MLflow
- **Forecasting Model**: Prophet
- **Core Language**: Python

## 4. How to Run This Project

### Prerequisites
1.  An Azure account with sufficient permissions.
2.  A GitHub account.
3.  Azure CLI installed locally.
4.  Terraform installed locally.

### Setup & Deployment
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/your-mlops-demand-forecasting.git
    cd your-mlops-demand-forecasting
    ```

2.  **Configure Azure Credentials for GitHub:**
    - Create an Azure Service Principal with `Contributor` role over your subscription.
    ```bash
    az ad sp create-for-rbac --name "github-actions-sp" --role "Contributor" --sdk-auth
    ```
    - Copy the resulting JSON object.
    - In your GitHub repository, go to `Settings > Secrets and variables > Actions` and create a new secret named `AZURE_CREDENTIALS`. Paste the JSON object as the value.

3.  **Customize Terraform Variables:**
    - Open `infrastructure/azure/variables.tf`.
    - It's highly recommended to change the `storage_account_name` to a globally unique value. The pipeline already appends the GitHub Run ID to ensure uniqueness during automated runs.

4.  **Push to GitHub:**
    - Commit and push your changes to the `main` branch. This will automatically trigger the GitHub Actions workflow defined in `.github/workflows/azure_mlops_pipeline.yml`.
    ```bash
    git add .
    git commit -m "Initial project setup"
    git push origin main
    ```

## 5. Project Results & Business Impact

* **Automation**: The pipeline reduces manual effort by **[Estimate % reduction, e.g., 95%]** for model retraining and deployment.
* **Forecast Accuracy**: The Prophet model achieved a Mean Absolute Error (MAE) of **[Find MAE from your MLflow run]**, providing reliable forecasts.
* **Scalability**: The solution is built on cloud services that can scale to handle terabytes of sales data across thousands of stores and products.
* **Business Value**: By improving forecast accuracy by an estimated **[Estimate %]**, this system can directly lead to a **[Estimate $$]** reduction in holding costs and a **[Estimate $$]** increase in revenue from prevented stockouts.
