# NeuralKarma OpenEnv Hackathon - Submission Guide

## CRITICAL TIMELINE

Submission Deadline: April 8, 2026, 11:59 PM IST

Current Status: Ready for submission

## Pre-Submission Checklist

- [x] inference.py in project root with correct logging format
- [x] Dockerfile that builds and runs successfully
- [x] openenv.yaml with all required fields
- [x] Professional README with no emojis
- [x] API endpoints: /api/reset, /api/step, /health
- [x] Three tasks with programmatic graders
- [x] Environment variables configured (API_BASE_URL, MODEL_NAME, HF_TOKEN)
- [x] No emojis in code or UI

## Step 1: Verify Local Setup

### 1.1 Install Python Dependencies

```bash
cd "/Users/laxmivarshitha/Documents/Hackathons/meta/NeuralKarma copy"

python3 --version  # Verify 3.10+

pip install -r requirements.txt
```

Expected: All packages install without errors

### 1.2 Start the Server Locally

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_huggingface_token_here"

python run.py --serve
```

Expected output:
```
[OK] NeuralKarma ML models loaded
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 1.3 Test Health Endpoint

Open another terminal:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "version": "1.0.0"}
```

### 1.4 Test OpenEnv Endpoints

```bash
# Test reset
curl -X POST http://localhost:8000/api/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "score_prediction"}'

# Test step
curl -X POST http://localhost:8000/api/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "score_prediction",
    "action": {"predicted_score": 75}
  }'
```

Expected: 200 status with JSON responses

### 1.5 Test Inference Script Locally

```bash
# Terminal 1: Keep server running
# (from step 1.2, server should be running on localhost:8000)

# Terminal 2: Override environment and run inference
export ENVIRONMENT_API="http://localhost:8000"
python inference.py

# Check for proper logging output:
# [START] task=... env=... model=...
# [STEP] step=... action=... reward=... done=... error=...
# [END] success=... steps=... rewards=...
```

Expected: Inference completes in 2-5 minutes with proper [START], [STEP], [END] format

---

## Step 2: Push to GitHub

### 2.1 Initialize Git Repository (if not already done)

```bash
cd "/Users/laxmivarshitha/Documents/Hackathons/meta/NeuralKarma copy"

git init
git add .
git commit -m "NeuralKarma OpenEnv Environment - Ready for submission"
```

### 2.2 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `neuralkarma-openenv` (or similar)
3. Description: "Ethical Impact Scoring Environment for OpenEnv Hackathon"
4. Add README.md (already in project)
5. Create repository

### 2.3 Push Code to GitHub

```bash
git branch -M main

git remote add origin https://github.com/YOUR_USERNAME/neuralkarma-openenv.git

git push -u origin main
```

Expected: Code successfully pushed to GitHub

---

## Step 3: Deploy to HuggingFace Spaces

### 3.1 Install HuggingFace CLI

```bash
pip install huggingface_hub

huggingface-cli login
# Paste your HF_TOKEN when prompted
```

### 3.2 Create HuggingFace Space

Method 1: Via Web UI (Faster)
1. Go to https://huggingface.co/spaces
2. Click "Create New Space"
3. Space name: `neuralkarma` (or similar, must be unique)
4. License: MIT
5. Space SDK: Docker
6. Visibility: Public
7. Click "Create Space"

Method 2: Via CLI
```bash
huggingface-cli repo create --type space neuralkarma --space-sdk docker
```

### 3.3 Clone and Setup Space Repository

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/neuralkarma

cd neuralkarma

# Copy your project files
cp -r "/Users/laxmivarshitha/Documents/Hackathons/meta/NeuralKarma copy"/* .

git add .

git commit -m "Deploy NeuralKarma to HuggingFace Spaces"

git push
```

### 3.4 Write Dockerfile for HF Space

Create `Dockerfile` (already included in your repo):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/ml/models

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini

CMD ["python", "run.py", "--serve"]
```

### 3.5 Monitor HF Space Build

1. Go to your Space URL: https://huggingface.co/spaces/YOUR_USERNAME/neuralkarma
2. Check the "Build" tab
3. Wait for Docker image to build (typically 5-10 minutes)
4. Expected: Space transitions to "Running" state

### 3.6 Verify Space is Running

```bash
curl https://YOUR_USERNAME-neuralkarma.hf.space/health
```

Expected response:
```json
{"status": "healthy", "version": "1.0.0"}
```

If not working yet, wait 2-3 more minutes or check logs in HF Space UI.

---

## Step 4: Pre-Submission Validation

### 4.1 Download Validation Script (Optional)

```bash
cd "/Users/laxmivarshitha/Documents/Hackathons/meta/NeuralKarma copy"

# The hackathon provides a validation script - download it
curl -fsSL https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/scripts/validate-submission.sh -o validate-submission.sh

chmod +x validate-submission.sh

# Run validation
./validate-submission.sh https://YOUR_USERNAME-neuralkarma.hf.space
```

### 4.2 Manual Validation Checks

1. **HF Space Ping Check**
   ```bash
   curl -I https://YOUR_USERNAME-neuralkarma.hf.space/health
   # Should return 200 OK
   ```

2. **Dockerfile Builds**
   ```bash
   cd "/Users/laxmivarshitha/Documents/Hackathons/meta/NeuralKarma copy"
   docker build -t neuralkarma-test .
   # Should complete without errors
   ```

3. **Inference Script Format**
   ```bash
   python inference.py 2>&1 | grep "^\[" | head -5
   # Should show [START], [STEP], [END] format
   ```

4. **OpenEnv YAML Valid**
   ```bash
   python -c "import yaml; yaml.safe_load(open('openenv.yaml'))"
   # Should execute without errors
   ```

---

## Step 5: Submit to Hackathon Platform

### 5.1 Navigate to Submission Portal

1. Go to the OpenEnv Hackathon platform
2. Sign in with your registered email
3. Find "Step 2: Submit your Assessment"
4. Click "Submit Assessment"

### 5.2 Fill Submission Form

**GitHub Repository URL (Required)**
```
https://github.com/YOUR_USERNAME/neuralkarma-openenv
```

**HuggingFace Space URL (Required)**
```
https://huggingface.co/spaces/YOUR_USERNAME/neuralkarma
```

### 5.3 Verify Space is Running Before Submit

**CRITICAL**: Your HF Space must be in "Running" state when you submit.

Check current state:
```bash
curl https://YOUR_USERNAME-neuralkarma.hf.space/health
```

If not running:
1. Go to https://huggingface.co/spaces/YOUR_USERNAME/neuralkarma
2. Click "Restart Space" if greyed out
3. Wait 3-5 minutes for restart
4. Verify /health endpoint returns 200

### 5.4 Submit Form

1. Paste both URLs into the submission form
2. Review all requirements checklist
3. Click "Submit"
4. Expected: Success confirmation page

---

## Step 6: Post-Submission Verification

### 6.1 Confirm Submission Recorded

1. Check your email for submission confirmation
2. Go back to hackathon platform
3. Verify your submission is listed with timestamp

### 6.2 Monitor Space During Evaluation

The evaluation system will:
1. Ping your /health endpoint
2. Call /api/reset and /api/step
3. Run your inference.py script
4. Verify output format

**Keep your space running** during the pre-submission window (27-28 March) for practice evaluation.

### 6.3 Handle Failures

If validation fails:

1. **Check error message** in hackathon platform feedback
2. **Common issues**:
   - Space not running: Restart via HF UI
   - Inference script fails: Check HF_TOKEN is set correctly
   - Wrong logging format: Verify [START], [STEP], [END] lines

3. **Fix and resubmit**: You can resubmit multiple times before deadline

---

## Important Notes

### Submission Deadline
- Final deadline: **April 8, 2026, 11:59 PM IST**
- Last resubmission must be before this time
- No extensions granted

### Space Maintenance
- Keep space running until evaluation completes (April 10)
- Do not delete or rename space after submission
- Do not modify code after submission

### Environment Variables
Your inference.py uses:
```
API_BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4o-mini"
HF_TOKEN = "your_huggingface_token_here"
```

These must be set when running both locally and in HF Space.

### Inference Requirements
- Must output [START], [STEP], [END] format exactly
- All field names must match exactly (case-sensitive)
- Rewards must be formatted to 2 decimal places
- Runtime must complete in under 20 minutes

---

## Quick Reference

### Local Testing
```bash
export HF_TOKEN="your_token"
python run.py --serve         # Terminal 1
python inference.py           # Terminal 2 (different terminal)
```

### GitHub Push
```bash
git add .
git commit -m "Message"
git push origin main
```

### HF Space Deploy
```bash
git clone https://huggingface.co/spaces/USERNAME/neuralkarma
cd neuralkarma
# Copy files
git add .
git commit -m "Deploy"
git push
```

### Verify Submission
```bash
curl https://USERNAME-neuralkarma.hf.space/health
```

---

## Support & Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'fastapi'"
**Solution**: Run `pip install -r requirements.txt`

### Issue: HF Space stuck building
**Solution**: 
1. Go to Space settings
2. Click "Restart Space"
3. Check build logs
4. Wait 5-10 minutes

### Issue: Inference script times out
**Solution**:
1. Check HF_TOKEN is correct
2. Reduce MAX_STEPS_PER_TASK in inference.py
3. Ensure models are pre-trained (run `python run.py --setup`)

### Issue: Wrong logging format
**Solution**: 
Check that output exactly matches:
```
[START] task=X env=Y model=Z
[STEP] step=N action=... reward=X.XX done=true|false error=...
[END] success=true|false steps=N rewards=...
```

---

## Final Checklist Before Submission

- [ ] inference.py in root directory
- [ ] Dockerfile builds without errors
- [ ] All dependencies in requirements.txt
- [ ] openenv.yaml is valid YAML
- [ ] Three task endpoints working (/api/reset, /api/step)
- [ ] /health endpoint returns 200
- [ ] No emojis in code
- [ ] README is professional and complete
- [ ] GitHub repo is public
- [ ] HF Space is running
- [ ] inferenceoutput has exact [START], [STEP], [END] format
- [ ] HF_TOKEN is correctly set
