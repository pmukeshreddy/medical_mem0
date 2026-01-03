# MedMem0 AWS Deployment Guide

## Prerequisites
- AWS Account
- GitHub repo with your code
- AWS CLI configured (optional but helpful)

---

## Part 1: Backend (AWS App Runner)

### Step 1: Push code to GitHub
Make sure your repo has this structure:
```
medical_mem0/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   ├── api/
│   ├── core/
│   └── models/
├── frontend/
└── Dockerfile  <-- add this to root
```

### Step 2: Create ECR Repository
```bash
aws ecr create-repository --repository-name medmem0-backend --region us-east-1
```

### Step 3: Build and Push Docker Image
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t medmem0-backend .

# Tag image
docker tag medmem0-backend:latest <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/medmem0-backend:latest

# Push image
docker push <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/medmem0-backend:latest
```

### Step 4: Create App Runner Service
1. Go to AWS Console → App Runner
2. Click "Create service"
3. Source: **Container registry** → Amazon ECR
4. Select your image: `medmem0-backend:latest`
5. Deployment: **Automatic** (for auto-deploy on push)
6. Configure service:
   - Service name: `medmem0-backend`
   - CPU: 1 vCPU
   - Memory: 2 GB
   - Port: 8000
7. Add Environment Variables:
   - PINECONE_API_KEY
   - PINECONE_INDEX_NAME
   - OPENAI_API_KEY
   - (see .env.example for all)
8. Create service

**Save the App Runner URL** (e.g., `https://xxxxx.us-east-1.awsapprunner.com`)

---

## Part 2: Frontend (AWS Amplify)

### Step 1: Go to AWS Amplify Console
1. AWS Console → Amplify
2. Click "New app" → "Host web app"

### Step 2: Connect GitHub
1. Select GitHub
2. Authorize AWS Amplify
3. Select your repo and branch (main)

### Step 3: Configure Build Settings
1. App name: `medmem0-frontend`
2. Root directory: `frontend` (if frontend is in subfolder)
3. Build settings will auto-detect Next.js

### Step 4: Add Environment Variable
1. Go to: App settings → Environment variables
2. Add:
   ```
   NEXT_PUBLIC_API_URL = https://xxxxx.us-east-1.awsapprunner.com
   ```
   (Use your App Runner URL from Part 1)

### Step 5: Update Build Settings (if needed)
In Amplify Console → Build settings, use:
```yaml
version: 1
frontend:
  phases:
    preBuild:
      commands:
        - npm ci
    build:
      commands:
        - npm run build
  artifacts:
    baseDirectory: .next
    files:
      - '**/*'
  cache:
    paths:
      - node_modules/**/*
```

### Step 6: Deploy
Click "Save and deploy"

---

## Part 3: Update CORS (Important!)

After getting your Amplify URL (e.g., `https://main.xxxxx.amplifyapp.com`), update backend CORS:

In `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://main.xxxxx.amplifyapp.com",  # Add your Amplify URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Rebuild and redeploy the backend.

---

## Final URLs
- **Frontend**: `https://main.xxxxx.amplifyapp.com`
- **Backend API**: `https://xxxxx.us-east-1.awsapprunner.com`
- **API Docs**: `https://xxxxx.us-east-1.awsapprunner.com/docs`

---

## Cost Estimate (Monthly)
- App Runner: ~$5-15 (auto-scales to zero when idle)
- Amplify: Free tier covers most demo usage
- **Total**: ~$5-20/month for demo

---

## Troubleshooting

**CORS errors?**
- Check backend CORS includes your Amplify URL

**502 errors on App Runner?**
- Check CloudWatch logs for backend errors
- Verify environment variables are set

**Frontend not connecting to backend?**
- Verify NEXT_PUBLIC_API_URL is set in Amplify
- Redeploy frontend after adding env var
