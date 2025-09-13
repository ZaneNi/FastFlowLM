---
title: Installation
nav_order: 1
has_children: false
---

## ⚙️ System Requirements

- 🧠 **Memory:** 32 GB RAM or higher recommended  
- ⚡ **CPU/NPU:** AMD Ryzen™ AI laptop with XDNA2 NPU  
- 🖥️ **OS:** Windows 11

> While FastFlowLM can run with 16 GB RAM, complex models (e.g., 3B or 8B) may require >= 32 GB for optimal performance and longer context length (more kv cache).

---

## 🚨 CRITICAL: NPU Driver Requirpement

You must have AMD NPU driver **version 32.0.203.258 or later** installed for FastFlowLM to work correctly.

- Check via:  
  **Task Manager → Performance → NPU**  
  or  
  **Device Manager → NPU**

🔗 [Download AMD Driver](https://www.amd.com/en/support)

---

## 💾 Installation (Windows)

A packaged FLM Windows installer is available here:  
[**flm-setup.exe**](https://github.com/FastFlowLM/FastFlowLM/releases/latest/download/flm-setup.exe)

If you see **"Windows protected your PC"**, click **More info**, then select **Run anyway**.

📺 [**Watch the quick start video**](https://www.youtube.com/watch?v=mYOfDNkyBII)

For version history and changelog, see the [release notes](https://github.com/FastFlowLM/FastFlowLM/releases/).

---

## 🚀 NPU Power Mode

By default, **FLM runs in `performance` NPU power mode**. You can switch to other NPU power modes (`powersaver`, `balanced`, or `turbo`) using the `--pmode` flag:

**CLI mode:**
```powershell
flm run gemma3:4b --pmode balanced
```

**Server mode:**
```powershell
flm serve gemma3:4b --pmode balanced
```

> ⚠️ Note: Using powersaver or balanced will lower NPU clock speeds and cause a significant drop in speed. For more details about NPU power mode, refer to the [AMD XRT SMI Documentation](https://ryzenai.docs.amd.com/en/latest/xrt_smi.html).


---

## 🧪 Quick Test (CLI Mode)

After installation, do a quick test to see if FastFlowLM is properly installed. Open **PowerShell**, and run a model in terminal (CLI mode):

```powershell
flm run llama3.2:1b
```

> **Notes:**
> - Internet access to HuggingFace is required to download the optimized model kernels.  
> - By default, models are stored in: `C:\Users\<USER>\Documents\flm\models\`  
> - During installation, you can select a different base folder (e.g., if you choose `C:\Users\<USER>\flm`, models will be saved under `C:\Users\<USER>\flm\models\`).
> - ⚠️ If HuggingFace is not directly accessible in your region, you can manually download the model (e.g., [hf-mirror](https://hf-mirror.com/models?search=fastflowlm)) and place it in the directory.
