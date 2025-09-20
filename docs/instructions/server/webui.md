---
title: Open WebUI + FLM
nav_order: 2
parent: Local Server (Server Mode)
---

# 📑 Table of Contents

- **[🧩 Run Open WebUI with FastFlowLM (Windows, YAML Method)](#-run-open-webui-with-fastflowlm-windows-yaml-method)**
- **[🧪 More Examples](#-more-examples)**
  - [🤖 Example: Multi Models Comparison Enabled by FLM Queuing (v0.9.10 or beyond)](#-example-multi-models-comparison-enabled-by-flm-queuing-v0910-or-beyond)

---

# 🧩 Run Open WebUI with FastFlowLM (Windows, YAML Method)

This guide walks you through using `docker-compose.yaml` to run Open WebUI connected to a local FastFlowLM instance on Windows.

---

## ✅ Prerequisites

1. [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
   - During installation, enable **WSL2 backend**
   - Reboot if prompted

2. [FastFlowLM](../../install.md)

---

## 📁 Step 1: Create Project Folder

Open PowerShell and run:

```powershell
mkdir open-webui && cd open-webui
```

This creates a clean workspace for your Docker setup.

---

## 📝 Step 2: Create `docker-compose.yaml`

Launch Notepad:

```powershell
notepad docker-compose.yaml
```

Paste the following:

```yaml
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    volumes:
      - open-webui-data:/app/backend/data
    environment:
      # Point WebUI to FLM's OpenAI-compatible server
      - OPENAI_API_BASE_URL=http://host.docker.internal:11434/v1
      - OPENAI_API_KEY=dummy-key

      # WebUI settings
      - WEBUI_AUTH=false
      - WEBUI_SECRET_KEY=dummysecretkey
      - ENABLE_TITLE_GENERATION=false
      - ENABLE_FOLLOW_UP_GENERATION=false
      - ENABLE_TAGS_GENERATION=false
      - ENABLE_RETRIEVAL_QUERY_GENERATION=false
      - ENABLE_IMAGE_PROMPT_GENERATION=false
      - ENABLE_WEB_SEARCH=false
      - ENABLE_SEARCH_QUERY_GENERATION=false
    restart: unless-stopped

volumes:
  open-webui-data:
```

---

## ▶️ Step 3: Launch the Open WebUI Container (in PowerShell)

```powershell
docker compose up -d
```
> It could take up to 1 min before you can access Open WebUI.

This starts the container in detached mode.  
You can check logs with:

```powershell
docker logs -f open-webui
```

---

## 🌐 Step 4: Access the WebUI (in Browser)

Open browser and go to:  
**http://localhost:3000**

You should now see the Open WebUI interface.

---

## 🧪 Step 5: Serve FastFlowLM with Model

```powershell
flm serve llama3.2:1b
```

You can now use `FastFlowLM` directly in Open WebUI.
> When switching models, it may take longer time to replace the model in memory.

---

## 🧼 Step 6: Stop or Clean Up (in PowerShell)

```powershell
docker compose stop
```

To **remove** it completely:

```powershell
docker compose down
```

This also removes the container but keeps persistent volume data.

or 

```powershell
docker compose down -v
```

This removes the container and persistent volume data.

---

## 🧼 Step 7: Update Open WebUI

```powershell
docker compose pull
```

---

## 🧠 Notes

- Want login? Set `WEBUI_AUTH=true`
- You must keep FastFlowLM server running
- For persistent chat history, the volume `openwebui-data` stores user data

---

> **Note (When using Open WebUI):**  
> The **Open WebUI** sends multiple background requests to the **server**.  
> To improve stability and performance, disable these in **Settings → Chat**:
> - **Title Auto-Generation**
> - **Follow-Up Auto-Generation**
> - **Chat Tags Auto-Generation**
> 
> Toggle them **off**, then refresh the page.

---

# 🧪 More Examples

## 🤖 Example: Multi Models Comparision Enabled by FLM Queuing (v0.9.10 or beyond)

A step-by-step guide to launching FastFlowLM and interacting with multiple models via Open WebUI.

---

### 🌐 Step 1: Run Open WebUI with FastFlowLM

Follow the quick setup at [here](https://docs.fastflowlm.com/instructions/server/).

---

### 🧩 Step 2:  Select and Add Models

At the top-right corner of the WebUI:

- Choose a model to begin (e.g., `llama3.2:1b`)
- Click **➕** to add other models, e.g.:
	- `qwen3:0.6b`
	- `gemma3:1b`

	You’ll now see several models listed. That means each one can answer your prompt.

---

### 💬 Step 3: Interact with Models

Type anything you're curious about in the input box.

⚠️ Please note:

- Each model will reply in sequences (not all at once)..
- The flm server dynamically loads each model based on your selection.

---

### 🎯 Step 4: Select or Merge

After receiving replies from multiple models, choose how you'd like to continue:

- ✅ **Use the Best Response**  
  Select the answer that best meets your expectations. That response will become the active context for your next question.

- 🔗 **Merge All Responses**  
  Combine insights from all models and continue the conversation using your preferred model. This lets you synthesize multiple perspectives into a unified thread.

---