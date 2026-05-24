# Azure Container Registry Deployment Steps

Follow these exact steps to build your Docker image locally and upload (push) it to your Azure Container Registry (ACR).

## Prerequisites
- You must have Docker installed and running on your machine.
- You must have the Azure CLI installed.

---

## Step 1: Create an Azure Container Registry (GUI)
1. Go to the [Azure Portal](https://portal.azure.com) in your web browser.
2. In the top search bar, search for **Container Registries** and select it (look for the blue hexagon icon).
3. Click **Create**.
4. Fill out the form:
   - **Subscription / Resource Group**: Select your active subscription (e.g., Azure for Students) and resource group.
   - **Registry name**: Type a globally unique name using only lowercase letters and numbers (e.g., `meridianfinanceacr`).
   - **Pricing plan**: Change this to **Basic** (cheaper, perfect for this project).
5. Click **Review + create**, then click **Create**.
6. Once deployed, click **Go to resource**. 
7. On the left menu under **Settings**, click **Access keys**, and toggle the **Admin user** setting to **Enabled**.

## Step 2: Log in to Azure via Terminal
Open your terminal in the root directory of this project (`meridian_finance`) and log in to your Azure account:

```bash
az login
```
*This will open a web browser for you to securely authenticate. Once logged in, return to the terminal.*

## Step 2: Log in to the Container Registry
Log in specifically to your newly created registry:

```bash
az acr login --name meridianfinanceacr
```

## Step 3: Build the Docker Image
Tell Docker to build your application image using the `Dockerfile` in this project, and tag it with your registry's exact login server address:

```bash
docker build -t meridianfinanceacr.azurecr.io/customer-intel-api:latest .
```
(Note: Do not forget the space and the `.` at the very end of the command, which tells Docker to build from the current directory).

## Step 4: Push the Image to Azure
Upload the newly built container image to your Azure registry:

```bash
docker push meridianfinanceacr.azurecr.io/customer-intel-api:latest
```

## Step 5: Deploy the Web App on Azure

The Docker image is now securely stored in your Azure Container Registry (`meridianfinanceacr`). However, the registry is only a storage vault—it does not actually run the code. To run your API and access its endpoints, you need to deploy the image to a hosting service like **Azure App Service (Web App for Containers)**.

You can do this using either the **Azure Portal (GUI)** or the **Azure CLI**.

### Option A: Deploy using the Azure Portal (GUI)

1. Open the [Azure Portal](https://portal.azure.com).
2. Click **Create a resource** in the top-left or search for **Web Apps** in the search bar and select it.
3. Click **Create** > **Web App**.
4. In the **Basics** tab, configure:
   - **Subscription**: Select your active Azure subscription.
   - **Resource Group**: Select your existing resource group.
   - **Name**: Choose a globally unique name for your API (e.g., `meridian-intel-api`). This will form your URL: `https://<name>.azurewebsites.net`.
   - **Publish**: Select **Container**.
   - **Operating System**: Select **Linux**.
   - **Region**: Select the same region where your ACR is located.
   - **Pricing Plan**: Choose a suitable pricing plan (e.g., Free F1 or Basic B1).
5. Click **Next: Container** at the bottom:
   - **Image Source**: Select **Azure Container Registry**.
   - **Registry**: Select `meridianfinanceacr`.
   - **Image**: Select `customer-intel-api`.
   - **Tag**: Select `latest`.
6. Click **Review + create**, then click **Create** and wait for deployment to finish.
7. Once deployed, click **Go to resource**.
8. Set up ports and keys:
   - On the left-hand menu, under **Settings**, click **Environment variables** (or **Configuration** in some layouts).
   - Add two new application settings:
     - **WEBSITES_PORT**: Set to `8000` (Crucial! This tells Azure App Service to route incoming traffic to your FastAPI container's Uvicorn port).
     - **GROQ_API_KEY**: Set to your actual Groq API key (Required for the RAG LLM model).
   - Click **Apply** or **Save** at the bottom/top of the settings page and confirm.

---

### Option B: Deploy using the Azure CLI (Terminal)

Alternatively, run these quick commands in your terminal to deploy from the command line:

1. **Create an App Service Plan** (Linux-based, Basic B1 tier):
   ```bash
   az appservice plan create \
     --name meridian-plan \
     --resource-group <Your-Resource-Group-Name> \
     --sku B1 \
     --is-linux
   ```

2. **Create the Web App** pointed at your ACR image:
   ```bash
   az webapp create \
     --name <Your-Web-App-Name> \
     --plan meridian-plan \
     --resource-group <Your-Resource-Group-Name> \
     --container-image-name meridianfinanceacr.azurecr.io/customer-intel-api:latest
   ```

3. **Configure App Settings** (expose port `8000` and pass the `GROQ_API_KEY`):
   ```bash
   az webapp config appsettings set \
     --name <Your-Web-App-Name> \
     --resource-group <Your-Resource-Group-Name> \
     --settings WEBSITES_PORT="8000" GROQ_API_KEY="<Your-Groq-API-Key>"
   ```

---

## Step 6: Access and Test the Endpoints

Once your Web App is running and the settings are applied:

1. **Find your URL**: Under the **Overview** section of your App Service in the Azure Portal, locate the **Default domain** (e.g., `https://meridian-intel-api.azurewebsites.net`).
2. **Interactive Docs (Swagger UI)**:
   Navigate to the following URL in your web browser:
   ```
   https://<Your-Web-App-Name>.azurewebsites.net/docs
   ```
   This will load the interactive FastAPI Swagger page where you can inspect and try out all the endpoints!
3. **Health Check Endpoint**:
   Check if the server is responsive and the ML/FAISS models loaded successfully:
   ```bash
   curl https://<Your-Web-App-Name>.azurewebsites.net/health
   ```
4. **Predict Conversion Endpoint**:
   Send a POST request to score a customer profile:
   ```bash
   curl -X POST "https://<Your-Web-App-Name>.azurewebsites.net/predict" \
        -H "Content-Type: application/json" \
        -d '{"age": 35, "job": "blue-collar", "marital": "married", "education": "basic.9y", "default": "no", "housing": "yes", "loan": "no", "contact": "cellular", "month": "may", "day_of_week": "fri", "duration": 150, "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent", "emp.var.rate": -1.8, "cons.price.idx": 92.893, "cons.conf.idx": -46.2, "euribor3m": 1.299, "nr.employed": 5099.1}'
   ```
5. **RAG Complaints Endpoint**:
   Send a query to the complaint assistant:
   ```bash
   curl -X POST "https://<Your-Web-App-Name>.azurewebsites.net/ask-complaints" \
        -H "Content-Type: application/json" \
        -d '{"question": "What is the policy on overdraft fees?", "filter_product": "Checking Account"}'
   ```
op