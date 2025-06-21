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
```mermaid
graph TD
    subgraph "Local / GitHub"
        A[Developer] -- Pushes Code --> G[GitHub Repo];
        G -- Triggers on Push --> GA[GitHub Actions CI/CD];
    end

    subgraph "Azure Cloud"
        subgraph "Provisioning (One-Time)"
            T[Terraform];
        end

        subgraph "Data & ML Pipeline (Automated)"
            SB[Azure Blob Storage];
            DB[Azure Databricks];
            AML[Azure Machine Learning Workspace];
            REG[AML Model Registry];
        end
    end

    %% Define Workflow
    GA -- "1. Runs terraform apply" --> T;
    T -- "Provisions" --> SB;
    T -- "Provisions" --> DB;
    T -- "Provisions" --> AML;

    GA -- "2. Uploads Raw Data" --> SB_Raw[raw-data container];
    
    GA -- "3. Triggers ML Pipeline in AML" --> AML;
    
    AML -- "Runs ETL Job on" --> DB;
    DB -- "Reads from" --> SB_Raw;
    DB -- "Writes Processed Data to" --> SB_Proc[processed-data container];
    
    AML -- "Runs Training Job" --> Train[Training Script on AML Compute];
    Train -- "Reads from" --> SB_Proc;
    Train -- "Uses MLflow for Tracking" --> AML;
    Train -- "Trains Prophet Model & Registers" --> REG;

    %% Style Definitions
    style G fill:#282C34,stroke:#FFF,stroke-width:2px,color:#FFF
    style GA fill:#2088FF,stroke:#FFF,stroke-width:2px,color:#FFF
    style T fill:#623CE4,stroke:#FFF,stroke-width:2px,color:#FFF
    style AML fill:#0078D4,stroke:#FFF,stroke-width:2px,color:#FFF
    style DB fill:#FF3600,stroke:#FFF,stroke-width:2px,color:#FFF
    style SB fill:#5BC5F2,stroke:#FFF,stroke-width:2px,color:#FFF

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